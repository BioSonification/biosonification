Audience: student scientific conference attendees, likely interdisciplinary, with limited time but interest in AI, bioinformatics, and computational creativity.
Objective: explain the research question, show the full reproducible pipeline, highlight the statistically supported result, and leave the audience with a clear sense that the project is scientifically honest, visually memorable, and promising for future work.
Narrative arc: from intrigue ("can biological sequences steer music generation?") to method (reproducible 5-stage pipeline), then to evidence (H1 supported, H2 not yet supported), and finally to research value and next steps.

Slide list:
1. Cover and hook.
The talk opens with the core idea of BioSonification as a bridge from biological sequence structure to symbolic music generation, framed as an AI research problem rather than an artistic gimmick.

2. Problem and hypotheses.
This slide states the gap in the field: many attractive demos, few reproducible pipelines with hypothesis testing. It introduces H1 and H2 in compact, memorable form.

3. End-to-end pipeline.
This is the central explanatory slide. It visualizes the 5 stages: FASTA -> bio-vector -> interpretable sonification -> paired MIDI dataset -> conditioned Transformer -> evaluation and MIDI outputs.

4. Data and experiment design.
This slide grounds the work in concrete numbers: 2000 biological sequences, 434 paired examples, split 303/65/66, vocabulary size 408, and explicit no-leakage control.

5. From biological features to musical control.
This slide explains how nucleotide frequencies, entropy, skew, GC content, and k-mer diversity become key, tempo, pitch range, scale type, and rhythm complexity. The message is interpretability before neural generation.

6. Model and experimental protocol.
This slide presents the conditioned Transformer and the comparison setup against baselines. It shows the model scale, training configuration, and the six generation regimes used in evaluation.

7. Results and hypothesis testing.
This is the evidence slide. It compares conditioned vs unconditional quality and shows that H1 is statistically supported, while H2 remains above chance but not significant.

8. Musical artifacts, conclusion, and outlook.
The final slide shows piano-roll examples and condenses the takeaways: conditioning helps, transfer is limited, and the pipeline is a solid platform for further research.

Source plan:
- Primary project sources: /Users/ivanvlasenko/Documents/diplom/biosonification/docs/thesis_chapters.md, /Users/ivanvlasenko/Documents/diplom/biosonification/docs/architecture_and_science.md, /Users/ivanvlasenko/Documents/diplom/biosonification/docs/code_walkthrough.md, /Users/ivanvlasenko/Documents/diplom/biosonification/docs/project_structure.md
- Conference framing and original thesis wording: /Users/ivanvlasenko/Downloads/МНСК тезис Власенко.pdf
- Quantitative experiment results: /Users/ivanvlasenko/Documents/diplom/biosonification/results/full_paired_run/final_report.json, /Users/ivanvlasenko/Documents/diplom/biosonification/results/full_paired_run/summary.txt, /Users/ivanvlasenko/Documents/diplom/biosonification/SCIENTIFIC_ARTICLE_AND_TALK_BASE.md
- Existing visual assets: /Users/ivanvlasenko/Documents/diplom/biosonification/results/full_paired_run/visualizations/bio_vectors_tsne.png, /Users/ivanvlasenko/Documents/diplom/biosonification/results/full_paired_run/visualizations/piano_roll_conditioned_0.png, /Users/ivanvlasenko/Documents/diplom/biosonification/results/full_paired_run/visualizations/piano_roll_unconditional_0.png

Visual system:
- Minimalist white presentation with warm white backgrounds, very light gray panels, thin graphite outlines, and restrained aqua/coral accents.
- Typography: Poppins for titles, Lato for body and captions.
- Mood: clean, scientific, airy, tactile, with subtle biological and musical motifs.
- Design rule: little text on slides, large numbers, one main chart or visual gesture per slide, and strong whitespace.

Imagegen plan:
- Each slide gets a dedicated text-free art-direction plate in a coherent bright visual system.
- Slide 1 sets the style with a pale, elegant fusion of DNA curves, note-like structures, and soft technical texture.
- Slides 2-8 vary composition while preserving the same palette and restrained scientific mood, leaving calm regions for editable titles, cards, and charts.

Asset needs:
- Use the generated art plates as background or side-panel atmosphere.
- Use native editable shapes for hypotheses, pipeline blocks, flow arrows, and summary cards.
- Use native editable charts for result comparisons.
- Use existing local raster assets only where they carry real experimental evidence, especially t-SNE and piano-roll examples.

Editability plan:
- All meaningful titles, subtitles, numbers, labels, hypotheses, pipeline stages, chart labels, and conclusion bullets will be authored as editable PowerPoint objects.
- Generated images remain decorative or atmospheric only.
- Speaker notes will hold the presenter guidance and source trace for each slide.
