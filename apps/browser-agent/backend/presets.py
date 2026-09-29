"""Example tasks shown on the start page.

Each one is a task of the 125-task real-website evaluation (evaluation/liveweb), quoted verbatim with its start page.
In that evaluation WebJev-35B-A3B completed all six and Jev 1.13 failed all six. Live websites change, so a rerun
can end differently.
"""

PRESETS = [
    {
        "id": "apple-macbook-air", "label": "Apple · MacBook Air specs", "task_id": "vts-021",
        "target_website": "https://www.apple.com/",
        "query": "On apple.com, open the Tech Specs page for the currently offered MacBook Air and leave the "
                 "specifications displayed.",
    },
    {
        "id": "mta-brooklyn-maps", "label": "MTA · Brooklyn maps", "task_id": "vts-019",
        "target_website": "https://new.mta.info/",
        "query": "On new.mta.info, find the list of Brooklyn neighborhood maps. Leave the Brooklyn neighborhood-map "
                 "listing open, with the Brooklyn map links displayed.",
    },
    {
        "id": "arxiv-paper-venue", "label": "arXiv · paper venue", "task_id": "vts-133",
        "target_website": "https://arxiv.org/",
        "query": "On arxiv.org, open the abstract page of the paper \"On the Sentence Embeddings from Pre-trained "
                 "Language Models\" and report the publication venue given in that page's Comments field.",
    },
    {
        "id": "wolfram-hilbert", "label": "Wolfram|Alpha · Hilbert matrix", "task_id": "vts-131",
        "target_website": "https://www.wolframalpha.com/",
        "query": "Calculate the determinant of a 6x6 Hilbert matrix.",
    },
    {
        "id": "mayo-infertility", "label": "Mayo Clinic · treatment page", "task_id": "vts-001",
        "target_website": "https://www.mayoclinic.org/",
        "query": "On mayoclinic.org, open the Diagnosis & treatment section of the Female infertility article.",
    },
    {
        "id": "govuk-visa", "label": "GOV.UK · visa checker", "task_id": "vts-051",
        "target_website": "https://www.gov.uk/",
        "query": "On gov.uk, use the Check if you need a UK visa service to check whether an American citizen with no "
                 "dual British or Irish citizenship needs a visa to work in healthcare in the UK for longer than 6 "
                 "months. Leave the completed checker result page open and return only a JSON object with the "
                 "boolean field \"visa_required\".",
    },
]

PRESET_BY_ID = {item["id"]: item for item in PRESETS}
