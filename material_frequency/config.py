"""Configuration: locked corpus, the role-based material lexicon, discovery
gazetteer, and target journals. Editing this file is how you 'revisit' after
discovery — promote a discovered material into the right role group and re-run
(re-runs are cache-served, so it's cheap).
"""

# --- OpenAlex polite pool -------------------------------------------------
# `mailto` sends NO email; it just opts requests into OpenAlex's faster, more
# stable "polite pool". Swap freely.
EMAIL = "s55751939@gmail.com"

# --- Locked corpus (plant-application hydrogel substrates) ----------------
# Gel-family context only; plant-application phrases only (no biomedical leakage).
CONTEXT = "(hydrogel OR aerogel OR cryogel OR xerogel)"
APP_PLANT = (
    '("seed germination" OR "plant growth" OR "plant tissue culture" OR micropropagation '
    'OR microgreen OR horticultural OR hydroponic OR soilless OR seedling OR "crop growth" '
    'OR agronomic OR "soil amendment" OR "soil conditioner" OR "plant cultivation")'
)
CORPUS = f"{CONTEXT} AND {APP_PLANT}"

# Supplementary "gelator pedigree" count for gellan: its real footprint is in
# tissue-culture vocabulary, NOT framed as a hydrogel (see memory gellan-vocabulary-gap).
GELLAN_PEDIGREE = (
    '("gellan gum" OR Phytagel OR Gelrite) AND ("plant tissue culture" OR micropropagation '
    'OR "tissue culture" OR "culture medium" OR explant OR callus)'
)

# Supplementary GELLING-AGENT corpus: a DIFFERENT corpus (a lens), NOT the headline.
# Swaps the gel-family CONTEXT for gelling-agent / tissue-culture framing while keeping
# the same plant gate (APP_PLANT). Used by footprint.py to re-count Role 1 gelators that
# the engineered-hydrogel corpus under-represents (e.g. gellan, agar). Not merged into
# the locked prevalence metric.
GELLING_FRAME = (
    '("gelling agent" OR "solidifying agent" OR "gelling agents" OR "solidifying agents" '
    'OR "tissue culture" OR micropropagation OR "culture medium" OR explant OR callus)'
)
GELLING_CORPUS = f"{GELLING_FRAME} AND {APP_PLANT}"

# --- The four selected building blocks ------------------------------------
SELECTED = {"gellan_gum", "alginate", "cmc", "pva"}


def _m(key, label, openalex, regex, selected=False):
    return {"key": key, "label": label, "openalex": openalex, "regex": regex,
            "selected": selected}


# --- Role-based lexicon ----------------------------------------------------
# `openalex`: phrases used to build server-side count queries (full names only;
#             NO bare abbreviations — they can't be word-boundary-matched server-side).
# `regex`:    case-insensitive local patterns (abbreviations allowed, the corpus
#             is already gated) used for discovery / representative labelling.
ROLES = [
    ("Role 1 - Specific calcium-ion junction zones (egg-box / ion-bridged helix)", [
        _m("alginate", "alginate",
           ["alginate", "sodium alginate", "alginic acid", "calcium alginate"],
           [r"algin", r"\balg\b"], selected=True),
        _m("gellan_gum", "gellan gum",
           ["gellan gum", "gellan", "Phytagel", "Gelrite", "Kelcogel"],
           [r"gellan", r"phytagel", r"gelrite", r"kelcogel"], selected=True),
        _m("pectin", "pectin", ["pectin", "low-methoxyl pectin"], [r"\bpectin"]),
    ]),
    ("Role 2 - Non-specific calcium-ion ionic crosslinking", [
        _m("cmc", "carboxymethyl cellulose",
           ["carboxymethyl cellulose", "carboxymethylcellulose", "cellulose gum",
            "sodium carboxymethyl cellulose"],
           [r"carboxymethyl[\s-]?cellulose", r"cellulose gum", r"\bcmc\b", r"\bnacmc\b"],
           selected=True),
        _m("polyacrylic_acid", "polyacrylic acid / polyacrylate",
           ["polyacrylic acid", "poly(acrylic acid)", "carbomer", "sodium polyacrylate",
            "polyacrylate"],
           # (?<!meth): count acrylate / acrylic acid but NOT methacrylate / methacrylic acid
           [r"polyacrylic", r"(?<!meth)acrylic[\s-]?acid", r"(?<!meth)acrylate",
            r"\bpaa\b", r"carbomer"]),
        _m("carboxymethyl_starch", "carboxymethyl starch", ["carboxymethyl starch"],
           [r"carboxymethyl starch", r"\bcms\b"]),
        _m("carboxymethyl_chitosan", "carboxymethyl chitosan",
           ["carboxymethyl chitosan", "carboxymethyl-chitosan"],
           [r"carboxymethyl[\s-]?chitosan", r"\bcmcs\b"]),
        _m("hyaluronic_acid", "hyaluronic acid",
           ["hyaluronic acid", "hyaluronan", "sodium hyaluronate"], [r"hyaluron"]),
        _m("gum_tragacanth", "gum tragacanth / karaya", ["gum tragacanth", "gum karaya"],
           [r"tragacanth", r"karaya"]),
        _m("lignosulfonate", "lignosulfonate", ["lignosulfonate", "sodium lignosulfonate"],
           [r"lignosulfonate", r"lignosulphonate"]),
    ]),
    ("Role 3 - Hydroxyl hydrogen-bonding glue", [
        _m("pva", "polyvinyl alcohol",
           ["polyvinyl alcohol", "poly(vinyl alcohol)", "polyvinylalcohol"],
           [r"poly\s?\(?vinyl alcohol\)?", r"polyvinylalcohol", r"\bpva\b"], selected=True),
        _m("peg", "polyethylene glycol / oxide",
           ["polyethylene glycol", "poly(ethylene glycol)", "polyethylene oxide",
            "poly(ethylene oxide)"],
           [r"poly\s?\(?ethylene glycol\)?", r"poly\s?\(?ethylene oxide\)?",
            r"\bpeg\b", r"\bpeo\b"]),
        _m("pvp", "polyvinylpyrrolidone",
           ["polyvinylpyrrolidone", "poly(vinylpyrrolidone)", "povidone"],
           [r"polyvinylpyrrolidone", r"poly\(vinylpyrrolidone\)", r"povidone", r"\bpvp\b"]),
        _m("cnf", "cellulose nanofiber",
           ["cellulose nanofiber", "cellulose nanofibers", "nanocellulose",
            "nanofibrillated cellulose", "cellulose nanofibril"],
           [r"cellulose nanofib", r"nanocellulose", r"nanofibrillated cellulose", r"\bcnf\b"]),
        _m("cnc", "cellulose nanocrystal",
           ["cellulose nanocrystal", "nanocrystalline cellulose", "cellulose whisker"],
           [r"cellulose nanocryst", r"nanocrystalline cellulose", r"cellulose whisker",
            r"\bcnc\b"]),
        _m("bacterial_cellulose", "bacterial cellulose",
           ["bacterial cellulose", "bacterial nanocellulose"], [r"bacterial (nano)?cellulose"]),
        _m("hec", "hydroxyethyl cellulose", ["hydroxyethyl cellulose"],
           [r"hydroxyethyl[\s-]?cellulose", r"\bhec\b"]),
        _m("hpc", "hydroxypropyl cellulose", ["hydroxypropyl cellulose"],
           [r"hydroxypropyl[\s-]?cellulose", r"\bhpc\b"]),
        _m("guar_gum", "guar gum", ["guar gum", "guar"], [r"\bguar\b"]),
        _m("locust_bean_gum", "locust bean gum", ["locust bean gum", "ceratonia"],
           [r"locust bean gum", r"\blbg\b", r"ceratonia"]),
        _m("gum_arabic", "gum arabic", ["gum arabic", "acacia gum", "arabic gum"],
           [r"gum arabic", r"acacia gum", r"arabic gum"]),
        _m("cashew_gum", "cashew gum", ["cashew gum"], [r"cashew gum", r"anacardium"]),
        _m("pullulan", "pullulan", ["pullulan"], [r"pullulan"]),
        _m("sericin", "silk sericin", ["sericin", "silk sericin"], [r"sericin"]),
    ]),
    ("Role 4 - Non-calcium gelation (thermal / cationic / covalent)", [
        _m("agar", "agar", ["agar", "agar-agar"], [r"\bagar\b", r"agar-agar"]),
        _m("agarose", "agarose", ["agarose"], [r"\bagarose\b"]),
        _m("gelatin", "gelatin", ["gelatin", "gelatine"], [r"\bgelatin"]),
        _m("collagen", "collagen", ["collagen"], [r"\bcollagen"]),
        _m("curdlan", "curdlan", ["curdlan"], [r"curdlan"]),
        _m("konjac", "konjac glucomannan", ["konjac", "glucomannan"],
           [r"konjac", r"glucomannan", r"\bkgm\b"]),
        _m("chitosan", "chitosan", ["chitosan"], [r"\bchitosan\b"]),
        _m("methylcellulose", "methylcellulose", ["methylcellulose", "methyl cellulose"],
           [r"(?<![a-z])methyl[\s-]?cellulose", r"\bmc\b"]),
        _m("hpmc", "hydroxypropyl methylcellulose",
           ["hydroxypropyl methylcellulose", "hypromellose"],
           [r"hydroxypropyl[\s-]?methylcellulose", r"hypromellose", r"\bhpmc\b"]),
        _m("carrageenan", "carrageenan",
           ["carrageenan", "kappa-carrageenan", "iota-carrageenan"], [r"carrageenan"]),
        _m("xanthan_gum", "xanthan gum", ["xanthan gum", "xanthan"], [r"\bxanthan"]),
        _m("silk_fibroin", "silk fibroin", ["silk fibroin", "fibroin"], [r"fibroin"]),
        _m("pnipam", "poly(N-isopropylacrylamide)",
           ["poly(N-isopropylacrylamide)", "N-isopropylacrylamide"],
           [r"isopropylacrylamide", r"\bpnipam\b"]),
        _m("fibrin", "fibrin", ["fibrin", "fibrinogen"], [r"\bfibrin"]),
    ]),
    ("Context - Incumbents (agronomy status quo, not selected)", [
        _m("polyacrylamide", "polyacrylamide",
           ["polyacrylamide", "poly(acrylamide)"],
           [r"poly\s?\(?acrylamide", r"\bpam\b", r"\bpaam\b"]),
        _m("superabsorbent", "superabsorbent polymer (SAP)",
           ["superabsorbent polymer", "super absorbent polymer"],
           [r"super\s?absorbent", r"\bsap\b"]),
        _m("starch", "starch", ["starch"], [r"\bstarch"]),
        _m("cellulose_generic", "cellulose (generic)", ["cellulose"], [r"\bcellulose"]),
    ]),
]

# Flat map key -> material for cross-role dedup of counts.
MATERIALS = {}
for _role, _mats in ROLES:
    for _mat in _mats:
        MATERIALS.setdefault(_mat["key"], _mat)


# --- Per-(material, DOI) recognition exclusions ----------------------------
# A regex can match a string that, in one specific paper, denotes a DIFFERENT
# compound (an abbreviation / substring collision the pattern can't disambiguate
# without over-blocking legitimate hits elsewhere). Suppress just those (key, doi)
# pairs. DOIs are bare (no scheme), matched case-insensitively.
RECOGNITION_EXCLUSIONS = {
    # 'PAA' here is phenylacetic acid (the released plant-growth regulator),
    # NOT polyacrylic acid.
    ("polyacrylic_acid", "10.1007/s10853-016-9775-0"),
    # 'CMCS' here is carboxymethyl starch(es); the paper contains no chitosan.
    ("carboxymethyl_chitosan", "10.1007/s10570-026-07077-1"),
}


# --- Discovery gazetteer ---------------------------------------------------
# Broad polymer / biomaterial / crosslinker vocabulary used to flag materials
# that appear in the corpus but are NOT yet in the role lexicon above.
# (Anything found here & absent from the lexicon -> reports/discovered_candidates.csv.)
GAZETTEER = [
    # polysaccharides / gums
    "dextran", "cellulose acetate", "ethyl cellulose", "hydroxypropyl cellulose",
    "gum ghatti", "gellan", "welan gum", "diutan gum", "tara gum", "cashew gum",
    "psyllium", "inulin", "levan", "scleroglucan", "schizophyllan", "beta-glucan",
    "chondroitin", "heparin", "fucoidan", "laminarin", "amylose", "amylopectin",
    "cyclodextrin", "lignin", "lignosulfonate", "hemicellulose", "arabinoxylan",
    # proteins
    "zein", "soy protein", "whey protein", "casein", "keratin", "elastin",
    "albumin", "gluten", "sericin",
    # synthetic polymers
    "polyethylenimine", "polyethyleneimine", "polylactic acid", "polycaprolactone",
    "pluronic", "poloxamer", "polyurethane", "polydopamine", "polyaniline",
    "polypyrrole", "poly(lactic-co-glycolic", "pegda", "gelma", "gelatin methacryloyl",
    "acrylamide", "acrylic acid", "methacrylate", "polyacrylonitrile",
    # nanofillers / inorganics / substrates
    "graphene", "graphene oxide", "carbon nanotube", "montmorillonite", "laponite",
    "halloysite", "silica", "hydroxyapatite", "bentonite", "kaolin", "zeolite",
    "vermiculite", "perlite", "biochar", "attapulgite", "clay",
    # crosslinkers / additives
    "calcium chloride", "glutaraldehyde", "genipin", "borax", "boric acid",
    "citric acid", "tannic acid", "epichlorohydrin", "polyphosphate",
]


# --- Target journals (soft preference for representative papers) -----------
# We do NOT filter the corpus by journal (would bias the denominator); we only
# flag representative papers that sit in these high-impact / on-topic venues.
TARGET_JOURNALS = [
    # polymer / hydrocolloid / biomaterials
    "carbohydrate polymers", "international journal of biological macromolecules",
    "food hydrocolloids", "biomacromolecules", "gels",
    "acs applied materials & interfaces", "acs applied materials and interfaces",
    "advanced functional materials", "advanced materials", "advanced healthcare materials",
    "acta biomaterialia", "biomaterials", "chemical engineering journal",
    "materials today", "acs nano", "acs sustainable chemistry & engineering",
    # plant / agriculture / food
    "nature plants", "plant cell, tissue and organ culture", "plant cell tissue and organ culture",
    "scientia horticulturae", "frontiers in plant science",
    "journal of agricultural and food chemistry", "food chemistry",
    "postharvest biology and technology", "agricultural water management",
    # space / controlled-environment agriculture
    "npj microgravity", "life sciences in space research", "astrobiology",
    "frontiers in astronomy and space sciences",
    # high-generalist
    "nature communications", "science advances", "nature", "science",
]
