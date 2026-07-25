"""Configuration for 08_material_frequency.ipynb: the locked-corpus search
query, the role-based material lexicon, and per-(material, DOI) recognition
exclusions.
"""

# Gel-family context only; plant-application phrases only (no biomedical leakage).
CONTEXT = "(hydrogel OR aerogel OR cryogel OR xerogel)"
APP_PLANT = (
    '("seed germination" OR "plant growth" OR "plant tissue culture" OR micropropagation '
    'OR microgreen OR horticultural OR hydroponic OR soilless OR seedling OR "crop growth" '
    'OR agronomic OR "soil amendment" OR "soil conditioner" OR "plant cultivation")'
)
CORPUS = f"{CONTEXT} AND {APP_PLANT}"


def _m(key, label, openalex, regex, selected=False):
    return {"key": key, "label": label, "openalex": openalex, "regex": regex,
            "selected": selected}


# `openalex`: phrases used to build server-side count queries.
# `regex`:    case-insensitive local patterns 

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
        _m("pullulan", "pullulan", ["pullulan"], [r"pullulan"]),
    ]),
    ("Other - Non-calcium gelation (thermal / cationic / covalent)", [
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

# Exclusion
RECOGNITION_EXCLUSIONS = {
    ("polyacrylic_acid", "10.1007/s10853-016-9775-0"),        # PAA = phenylacetic acid
    ("carboxymethyl_chitosan", "10.1007/s10570-026-07077-1"), # CMCS = carboxymethyl starch
    ("cnf", "10.1007/s10853-026-12465-w"),                    # acrylic-acid-grafted CNF
    ("cnf", "10.1002/adfm.202506427"),                        # grade unspecified
    ("cnf", "10.1007/s10924-023-03103-6"),                    # grade unspecified
    ("cnf", "10.1007/s10570-026-07077-1"),                    # false positive
}
