"""Build a printable speaker script for the deck.

Writes ``results/RFDT_speaking_notes.pdf``: one page per slide, each carrying a
thumbnail of the slide so you can see where you are, then what to say.

Stage directions are set apart in bold small caps so they read differently from
the words themselves.  Anything in a line beginning with a marker like "POINT
AT" or "IF ASKED" is an instruction to you, not something to say aloud.

Kept local rather than committed, like the rest of the slide tooling.
"""

from __future__ import annotations

import os
import sys
import textwrap
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages                  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
OUTPUT = os.path.join(RESULTS, "RFDT_speaking_notes.pdf")

#: Markers that flag a stage direction rather than words to speak.
DIRECTIONS = ("OPEN.", "TIMING.", "PAUSE", "POINT AT", "IF ASKED", "SAY ",
              "KEY LINE", "THE TAKEAWAY", "MOVE THROUGH", "THIS SETS UP",
              "WHY THAT MATTERS", "THAT IS THE HONEST")

#: (slide number, image file or None, heading, notes)
SCRIPT: List[Tuple[int, str, str, str]] = [
 (1, None, "Title",
  """This is a radio propagation simulator I wrote from scratch, applied to a robot moving around an indoor space, and used to work out how much the material of the walls actually matters.

One thing to get out of the way first. This is not Sionna. I wrote the ray tracer myself, on PyTorch, because the method I am implementing is about getting correct gradients with respect to the geometry of a scene, and the paper shows Sionna's gradients break down exactly there. So wrapping an existing tracer would not have given me the thing I wanted to test.

The physics is checked against closed-form results, twenty nine automated tests, and I will come back to that near the end.

TIMING. About thirty seconds. Do not linger here."""),

 (2, "slides/A1_environment.png", "The environment",
  """This is the scene. A six by five metre room, two point eight metres high, twenty four surfaces. Access point on the ceiling in that corner, and the robot carries the receiver at about waist height, ninety centimetres, along the route marked in blue.

POINT AT THE PARTITION, the vertical slab at three metres. That is the important piece. It runs from the wall about two thirds of the way across and stands two point four metres tall, so it puts the far half of the room in shadow, but it leaves a gap the robot can drive around.

That is deliberate. It means one single route takes the robot through clear line of sight, then across a shadow boundary, then into a region with no direct path at all, and back out. Those three regimes behave completely differently and I wanted all of them in one run.

Materials are colour coded. Concrete walls, wood floor, ceiling tile, a metal cabinet and a wooden table.

IF ASKED why those materials, they are just the defaults, and the whole of slide eleven is about varying them."""),

 (3, "slides/A2_ray_trace.png", "Traced propagation paths",
  """These are the actual propagation paths the simulator found at one position. Sixty eight of them. I am only drawing the fourteen strongest, because all sixty eight at once is unreadable.

Line thickness is path strength, so the picture shows you what matters, not just what exists. Orange is the direct path, blue is a single bounce, purple is two bounces, green is diffracted around a corner.

Worth saying plainly, these are the traced polylines, not an artist's impression. The length of each line you see equals the path length the simulator reports, exactly. The dots mark where each bounce happens.

POINT AT THE SIDE VIEW on the right. That earns its place because it shows paths clearing obstacles vertically, which you cannot see from overhead. A path can look blocked from above and actually be going over the top of the furniture."""),

 (4, "slides/A3_route_regimes.png", "The same link in three regimes",
  """Same link, three positions along the route.

On the left, clear line of sight, and the direct path carries almost everything.

In the middle, near the shadow boundary, the direct path is weakening.

On the right, deep in the shadow behind the partition, and now the channel is carried entirely by reflections and by energy diffracting around edges.

Notice the numbers under each panel. Received power and delay spread both change, and the character of the link changes completely, even though it is the same transmitter and the same receiver.

THIS SETS UP the rest of the talk. That shadow boundary in the middle is exactly where a conventional simulator has a problem, and I come back to why on slide twelve."""),

 (5, "slides/B1_paths_1.png", "One path",
  """Now the part I found most useful for actually understanding what is going on.

I am going to take that same link and build it up one path at a time, showing three things together each time. The impulse response top right, the amplitude across frequency in the middle, and the phase at the bottom.

Start with one path. Just the direct path, nothing else.

The impulse response is a single spike, which makes sense, one path arriving at one time. The amplitude is completely flat, so every frequency is treated identically. And the phase is a straight ramp, whose slope is exactly the propagation delay.

That is a clean channel. Nothing interesting happens to a signal going through it.

PAUSE HERE. This is the baseline everything else gets measured against."""),

 (6, "slides/B2_paths_2.png", "Two paths",
  """Add one more path. A single bounce off a wall.

The impulse response now has two spikes, the second arriving later and weaker.

And look what happens to the amplitude. The two copies interfere and you get a regular comb of notches. Six point three decibels from peak to trough, from adding one reflection.

The spacing of those notches is not arbitrary. It is one divided by the difference in arrival time between the two paths, and that is annotated on the slide. I checked it against theory and it comes out within a fraction of a percent, which I show on slide ten.

The phase has also stopped being a straight line. It now wobbles around the ramp.

KEY LINE. One reflection is enough to put a six decibel hole in your channel."""),

 (7, "slides/B3_paths_3.png", "Three paths",
  """Three paths, and the comb stops being regular.

That is because you now have three different arrival times beating against one another rather than two, so there is no single spacing any more.

Thirteen decibels peak to trough now, up from six.

MOVE THROUGH THIS ONE QUICKLY. It is a stepping stone, the interesting jump is next."""),

 (8, "slides/B4_paths_68.png", "All sixty eight paths",
  """And here is everything. All sixty eight paths.

The impulse response is now what you would actually measure in a real room. A strong direct arrival, then a cluster of reflections, then a long tail of weaker energy arriving later.

The amplitude is deeply and irregularly notched, nearly thirty decibels peak to trough. And the phase has torn away from the straight ramp entirely, with sharp excursions at every notch.

This is what a real indoor channel looks like. If you are designing a radio to work in this room, this is the thing it has to cope with.

IF ASKED why the notches matter, a notch means that particular frequency is essentially unusable at that position, and if the robot moves slightly the notches move with it."""),

 (9, "slides/B5_summary.png", "What each added path does",
  """This is the summary of that sequence, and it is the single most useful slide in the deck.

Three quantities as I added paths. On the left, frequency selectivity, how deeply notched the channel is. In the middle, delay spread, how smeared in time it is. On the right, received power.

Look at the difference. Selectivity goes from zero to twenty nine decibels. Delay spread from zero to three point one nanoseconds. But received power moves by under three decibels across the entire sequence.

THE TAKEAWAY LINE. Multipath changes the shape of the channel, not its level.

The reason is simple. Most of your received power is the direct path, and the direct path never touches a wall. So if you only measure average signal strength you will conclude the room barely matters, and that conclusion would be wrong. The other two charts are why.

PAUSE HERE. This is the point of the first half of the talk."""),

 (10, "exp5_frequency_response.png", "Frequency response and validation",
  """Now, how do I know any of this is right.

This is the frequency domain, and it is where I check the simulator against closed-form physics rather than against itself.

Top left. For two paths, theory says the notch spacing should be the speed of light divided by the path length difference. Theory says two hundred and seventy seven point four nine megahertz. The simulator gives two hundred and seventy seven point six seven. That is six hundredths of one percent.

Also on that panel, the direct path alone has no notches at all, and falls by exactly the amount free space spreading predicts across the band.

Top right, phase and group delay. For a single path the group delay comes out equal to distance over the speed of light, agreeing to fifteen decimal places.

Bottom right is where I correct my own earlier numbers. I had been reporting coherence bandwidth using the standard rule of thumb from delay spread. Measuring it properly here shows the rule is fine for metal walls but understates the true value by roughly a factor of four for concrete and foam. The ordering across materials survived, the absolute numbers did not, so I relabelled them.

SAY THAT LAST PART OUT LOUD. Owning it is stronger than hiding it."""),

 (11, "exp2_material_sweep.png", "Material sweep",
  """This is the material result. Eleven materials from the international standard database, with the robot driving the full route.

Bottom left is the cause. That is just the reflectivity of each material against angle. Metal reflects everything, foam reflects almost nothing, everything else in between.

The top row is the effect. And the surprise is top left. At five gigahertz the received power barely changes across all eleven materials, only about four decibels, for the reason we just saw.

But the middle and right panels show the channel shape changing enormously, and in exact order of reflectivity. A metal room behaves like a tiled bathroom, echoes louder than the direct signal. A foam room behaves like a recording studio.

Bottom middle is the practical one. Signal strength as the robot drives, four materials, and the lines basically move together. The big dips are the robot going behind the partition. Twenty five decibels of variation along the route against four decibels between materials, so where the robot is matters far more than what the walls are made of.

Bottom right is the other side of it, energy going through a wall rather than bouncing off. Concrete costs twelve decibels at five gigahertz but eighty one at sixty. That gap is essentially why millimetre wave does not work through walls.

IF ASKED which material is best, it depends what you are optimising. Reflective walls help coverage in shadowed areas but wreck delay spread. There is no single winner."""),

 (12, "slides/D1_method_of_images.png", "How the second path is found",
  """Two slides on how this works underneath, because the method is the contribution here, not the numbers.

Finding a reflected path. Mirror the receiver through the plane of the wall, draw a straight line from the transmitter to that mirror image, and wherever that line crosses the wall is your bounce point. It satisfies the law of reflection by construction, and the total path length is just the straight line distance to the image. You can see the dashed construction line passing exactly through the traced bounce point.

Now the detail that matters, and this is the whole reason I built this rather than using something off the shelf.

That crossing is computed on the infinite plane, not on the finite wall. So a bounce point always exists, and it moves smoothly when the wall moves. Whether it actually lands on the real wall is applied afterwards, as a smooth weight rather than a yes or no test.

WHY THAT MATTERS. With a yes or no test, asking "would the signal improve if this wall were slightly bigger" gives you an answer of exactly zero, everywhere, because a step function has no slope. Which means you can never learn the size of a reflector from measurements. With the smooth version you get a usable answer, and I verified it matches brute force to four decimal places."""),

 (13, "slides/D2_higher_order.png", "Third path and beyond",
  """Same idea extended.

On the left, two bounces. You mirror the receiver through both wall planes in turn, then sweep forward finding each bounce point. It generalises to any number of bounces, though the cost grows.

On the right, diffraction. When a signal bends around the edge of an object, the bend point is the one that minimises the total path length along that edge. That has a closed-form solution, so there is no iterative search and the gradient stays clean.

MOVE THROUGH THIS QUICKLY unless someone asks. The important idea was on the previous slide."""),

 (14, None, "Validation and limitations",
  """What I have checked, and what I have not.

On the left, five checks against closed-form physics. Free space transmission agrees with the standard equation to under a thousandth of a decibel. A two ray ground reflection agrees with the analytic answer to under five hundredths. Reciprocity, meaning swapping the transmitter and the receiver, holds to thirteen decimal places, and that one is brutal because almost any bookkeeping error breaks it. Notch spacing to six hundredths of a percent. And the gradients match brute force finite differences to nine decimal places.

Twenty nine automated tests, they run in three seconds, and the pipeline refuses to generate any results if one of them fails.

On the right, what this does not do. Single polarisation. First order diffraction only. Approximate values for four of the materials, labelled everywhere they appear. Small scenes only, this is not a replacement for a GPU tracer on a whole building.

And the important one, third down. No measured ground truth. Everything here is validated against theory and internal consistency, not against a real measurement campaign. So the honest claim is that the physics is implemented correctly, not that it has been confirmed in a real room.

THAT IS THE HONEST STOPPING POINT. If nobody asks what is next, stop here."""),

 (15, "slides/E1_dloc_validation.png", "Next: validating against measured data",
  """This answers the bullet I just flagged.

The DLoc dataset from UCSD gives a hundred and five thousand measured WiFi channels at five gigahertz, collected by a mapping robot in two real spaces. That is exactly the measured ground truth this work is missing, so the plan is to simulate those spaces and compare.

On the left is the space as published with the dataset, and underneath it the floor plan I extracted from the robot's own occupancy map. So the geometry is coming from their data, not from me drawing a room.

The material assumptions come from that photograph. Drywall throughout, plasma screens on two opposite walls, and a metal column beside the fourth access point.

Now the part I want to be careful about. Raw channel measurements cannot be compared directly against a simulator. There is an unknown timing offset on every packet, the phase is randomised packet to packet, and the amplitude is in arbitrary units because of automatic gain control. So absolute delay, phase and magnitude are all off the table.

What is fair to compare is received signal strength, the shape of the response across frequency, spatial correlation along the route, and angle of arrival. I have written that down, with pass and fail thresholds, before opening the measurements.

SAY THIS EXPLICITLY. There is no predicted result on this slide, deliberately. If I committed to an outcome before running the test, the test would not mean anything. The protocol is written so that the simulator failing to match is as informative as it succeeding, and if it does fail, that motivates building something better for sub six gigahertz.

Currently blocked on the consent form for the measurements, and on one scale factor that has to come from the data."""),
]


def is_direction(line: str) -> bool:
    """True when a line is a stage direction rather than words to speak."""
    return any(line.lstrip().startswith(d) for d in DIRECTIONS)


def _layout(notes: str, wide: bool = False) -> List[Tuple[str, bool]]:
    """Wrap the script into display lines, tagging each as speech or direction.

    A blank entry marks the gap between paragraphs.
    """
    out: List[Tuple[str, bool]] = []
    for para in notes.strip().split("\n\n"):
        para = " ".join(para.split())
        d = is_direction(para)
        for line in textwrap.wrap(para, width=(74 if d else 78)):
            out.append((line, d))
        out.append(("", False))
    return out[:-1] if out else out


def draw_page(pdf: PdfPages, number: int, image: str, heading: str,
              notes: str) -> None:
    """Render one slide's script, continuing onto further pages if needed.

    Earlier versions silently dropped anything that ran past the bottom of the
    page, which lost a stage direction on the two longest slides.  Overflow now
    continues onto a page marked "continued" instead of disappearing.
    """
    lines = _layout(notes)
    thumb = os.path.join(RESULTS, image) if image else None
    has_thumb = bool(thumb and os.path.exists(thumb))

    bottom = 0.055

    # Try progressively tighter layouts and take the first that fits on one
    # page.  A slide that spills two lines onto a continuation is worse to
    # speak from than one slightly denser page, and the earlier fixed-shrink
    # version still orphaned a line on the longest slides.
    LAYOUTS = [
        # thumbnail height, speech leading, direction leading, paragraph gap, font
        (0.240, 0.0245, 0.0225, 0.0140, 11.5),
        (0.205, 0.0238, 0.0218, 0.0130, 11.0),
        (0.175, 0.0228, 0.0209, 0.0120, 10.5),
        (0.150, 0.0218, 0.0200, 0.0110, 10.0),
        (0.130, 0.0208, 0.0191, 0.0100, 9.5),
    ]
    thumb_h, step_speech, step_dir, gap, font = LAYOUTS[0]
    for cand in LAYOUTS:
        th, ss, sd, gp, ft = cand
        top = (0.905 - th - 0.03) if has_thumb else 0.905
        need = sum(gp if t == "" else (sd if d else ss) for t, d in lines)
        if need <= top - bottom:
            thumb_h, step_speech, step_dir, gap, font = cand
            break
        thumb_h, step_speech, step_dir, gap, font = cand

    page = 0
    idx = 0

    while idx < len(lines) or page == 0:
        fig = plt.figure(figsize=(8.27, 11.69))          # A4 portrait
        cont = " (continued)" if page else ""
        fig.text(0.07, 0.965, f"Slide {number}{cont}", fontsize=11,
                 color="#2e86c1", fontweight="bold")
        fig.text(0.07, 0.938, heading, fontsize=17, fontweight="bold",
                 color="#1a1a1a")
        fig.add_artist(plt.Line2D([0.07, 0.93], [0.928, 0.928],
                                  color="#cccccc", lw=1.0))

        y = 0.905
        if has_thumb and page == 0:
            ax = fig.add_axes([0.07, 0.90 - thumb_h, 0.86, thumb_h])
            ax.imshow(plt.imread(thumb))
            ax.axis("off")
            y = 0.90 - thumb_h - 0.03

        while idx < len(lines):
            text, direction = lines[idx]
            step = gap if text == "" else (step_dir if direction else step_speech)
            if y - step < bottom:
                break
            if text:
                fig.text(0.07, y, text,
                         fontsize=(font - 1.0) if direction else font,
                         color="#8c2020" if direction else "#1a1a1a",
                         fontweight="bold" if direction else "normal",
                         family="sans-serif")
            y -= step
            idx += 1

        # do not start a continuation page with a blank separator
        while idx < len(lines) and lines[idx][0] == "":
            idx += 1

        fig.text(0.5, 0.028, f"{number} of {len(SCRIPT)}", fontsize=9,
                 color="#999999", ha="center")
        pdf.savefig(fig)
        plt.close(fig)
        page += 1
        if idx >= len(lines):
            break


def draw_cover(pdf: PdfPages) -> None:
    """Opening page: how to read the script, and the running order."""
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.07, 0.94, "Speaking notes", fontsize=26, fontweight="bold")
    fig.text(0.07, 0.905, "A differentiable RF simulator for indoor robotics",
             fontsize=13, color="#555555")
    fig.add_artist(plt.Line2D([0.07, 0.93], [0.893, 0.893], color="#cccccc"))

    fig.text(0.07, 0.855, "How to read this", fontsize=13, fontweight="bold",
             color="#2e86c1")
    for i, line in enumerate(textwrap.wrap(
            "Plain text is what to say. Lines in bold red are stage directions "
            "for you, not words to speak: where to point, when to pause, and "
            "what to say if a particular question comes up. One page per "
            "slide, with a thumbnail so you can see where you are.", 76)):
        fig.text(0.07, 0.822 - i * 0.024, line, fontsize=11.5)

    fig.text(0.07, 0.71, "Running order", fontsize=13, fontweight="bold",
             color="#2e86c1")
    y = 0.678
    for n, _, head, notes in SCRIPT:
        words = len(notes.split())
        fig.text(0.07, y, f"{n:2d}.", fontsize=11, color="#999999")
        fig.text(0.115, y, head, fontsize=11.5)
        fig.text(0.86, y, f"~{max(20, round(words / 140 * 60 / 5) * 5)}s",
                 fontsize=10, color="#777777", ha="right")
        y -= 0.0265

    total = sum(len(n.split()) for _, _, _, n in SCRIPT) / 140
    fig.text(0.07, y - 0.02,
             f"Full deck at a steady pace: about {total:.0f} minutes of "
             f"speaking, before questions.", fontsize=11.5, color="#555555")
    fig.text(0.07, y - 0.055,
             "Short version: slides 1, 2, 5, 6, 8, 9, 15.",
             fontsize=11.5, color="#555555")
    pdf.savefig(fig)
    plt.close(fig)


def main():
    """Write the speaker-notes PDF."""
    os.makedirs(RESULTS, exist_ok=True)
    with PdfPages(OUTPUT) as pdf:
        draw_cover(pdf)
        for number, image, heading, notes in SCRIPT:
            draw_page(pdf, number, image, heading, notes)
    total = sum(len(n.split()) for _, _, _, n in SCRIPT) / 140
    print(f"wrote {os.path.relpath(OUTPUT, ROOT)}  "
          f"({len(SCRIPT) + 1} pages, {os.path.getsize(OUTPUT)/1e6:.1f} MB)")
    print(f"  about {total:.0f} minutes of speaking across {len(SCRIPT)} slides")


if __name__ == "__main__":
    main()