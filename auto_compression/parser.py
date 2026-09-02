import re
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from scipy.signal import find_peaks


@dataclass
class SpecimenData:
    sample_name: str
    specimen_num: int
    strain: pd.Series  # X (%)
    stress: pd.Series  # Y (kPa)
    height_mm: float | None = None  # specimen height (continuous path only)


def extract_sample_name(filepath: Path) -> str:
    with open(filepath, 'r') as f:
        first_line = f.readline()
    match = re.search(r'SID,"([^"]+)"', first_line)
    return match.group(1) if match else filepath.stem


def find_data_sections(filepath: Path) -> list[tuple[int, int]]:
    """Find line indices where each specimen's data section starts."""
    sections = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        if re.match(r'^\d+,Time,Displacement,Force,Compressive stress,Compressive strain', line):
            specimen_num = int(line.split(',')[0])
            sections.append((i, specimen_num))
    
    return sections


def parse_specimen_data(filepath: Path, start_line: int, end_line: int | None) -> tuple[pd.Series, pd.Series]:
    """Parse strain and stress data from a specimen section."""
    df = pd.read_csv(
        filepath,
        skiprows=start_line + 2,  # skip header and units rows
        nrows=(end_line - start_line - 2) if end_line else None,
        header=None,
        usecols=[4, 5],  # stress (kPa), strain (%)
        names=['stress', 'strain']
    )
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    return df['strain'], df['stress']


def parse_csv(filepath: Path) -> list[SpecimenData]:
    """Parse a CSV file and return list of SpecimenData objects."""
    sample_name = extract_sample_name(filepath)
    sections = find_data_sections(filepath)
    
    if not sections:
        raise ValueError(f"No data sections found in {filepath}")
    
    specimens = []
    for i, (start_line, specimen_num) in enumerate(sections):
        end_line = sections[i + 1][0] if i + 1 < len(sections) else None
        strain, stress = parse_specimen_data(filepath, start_line, end_line)
        
        specimens.append(SpecimenData(
            sample_name=sample_name,
            specimen_num=specimen_num,
            strain=strain.reset_index(drop=True),
            stress=stress.reset_index(drop=True)
        ))
    
    return specimens


def parse_folder(folder_path: Path) -> list[SpecimenData]:
    """Parse all CSV files in a folder."""
    all_specimens = []
    for csv_file in sorted(folder_path.glob('*.csv')):
        all_specimens.extend(parse_csv(csv_file))
    return all_specimens


# ---------------------------------------------------------------------------
# Continuous-recording format ("Robustness Test - Compression (Second Test).csv")
#
# Unlike the per-specimen Instron exports above (one ``SID,"..."`` block with
# repeated ``N,Time,Displacement,...`` section headers), this file is a single
# uninterrupted recording in which many compression events run back-to-back.
# There are no section-header rows to split on, so the section-based parser
# cannot read it. Segment by detecting the displacement peak of each event.
#
# The file carries engineering stress over the true specimen cross-section
# (``SPECIMEN_AREA_MM2``). It has no strain column: the instrument's own strain
# channel was referenced to the fixture gap rather than to the specimen, and a
# specimen-referenced strain cannot be tabulated per row because it is defined
# only inside a loading event. It is derived per event below.
# ---------------------------------------------------------------------------

TIME_COL = 'Time (s)'
DISP_COL = 'Displacement (mm)'
FORCE_COL = 'Force (N)'
STRESS_COL = 'Compressive stress (kPa)'  # Y (kPa)

# Test-setup geometry. The specimen area is already applied in STRESS_COL; the
# gap and holder heights are needed here because specimen height is not recorded.
SPECIMEN_AREA_MM2 = 14.0 * 14.0  # true specimen cross-section
FIXTURE_GAP_MM = 20.0            # platen separation at zero displacement
HOLDER_HEIGHT_MM = 2.5           # holder beneath every specimen
CONTACT_FORCE_N = 0.05           # force defining first platen-specimen contact


def load_continuous_csv(filepath: Path) -> pd.DataFrame:
    """Load a continuous multi-event recording.

    All columns are coerced to numeric and rows with missing displacement or
    stress are removed.
    """
    df = pd.read_csv(filepath)
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.dropna(subset=[DISP_COL, STRESS_COL]).reset_index(drop=True)


def segment_loading_events(
    df: pd.DataFrame,
    sample_name: str = 'continuous',
    *,
    peak_height: float = 6.0,
    peak_distance: int = 100,
    peak_prominence: float = 4.0,
    contact_force: float = CONTACT_FORCE_N,
) -> list[SpecimenData]:
    """Split a continuous recording into per-event loading branches.

    Each compression event ramps the crosshead down to a peak displacement then
    unloads. We locate the displacement peaks, walk backward from each peak to
    first contact (last sample before force rises above ``contact_force``), and
    return the loading branch (contact -> peak). The unloading branch is
    intentionally discarded.

    Specimen height is not recorded, so it is reconstructed per event from the
    contact displacement -- the specimen occupies whatever is left of the
    fixture gap once the holder and the travel to contact are subtracted::

        height_mm = FIXTURE_GAP_MM - HOLDER_HEIGHT_MM - displacement_at_contact

    Strain is compressive strain referenced to that height, so every event
    starts at ``(0, ~0)``.

    ``specimen_num`` is the 1-based event index in acquisition order.
    """
    disp = df[DISP_COL].values
    force = df[FORCE_COL].values
    stress = df[STRESS_COL].values
    peaks, _ = find_peaks(disp, height=peak_height,
                          distance=peak_distance, prominence=peak_prominence)

    events = []
    for event_num, p in enumerate(peaks, 1):
        i = p
        while i > 0 and force[i] >= contact_force:
            i -= 1
        i_start = i + 1
        d0 = float(disp[i_start])
        height = FIXTURE_GAP_MM - HOLDER_HEIGHT_MM - d0
        strain = (disp[i_start:p + 1] - d0) / height * 100.0
        events.append(SpecimenData(
            sample_name=sample_name,
            specimen_num=event_num,
            strain=pd.Series(strain).reset_index(drop=True),
            stress=pd.Series(stress[i_start:p + 1]).reset_index(drop=True),
            height_mm=height,
        ))
    return events


def parse_continuous_csv(filepath: Path, **kwargs) -> list[SpecimenData]:
    """Load a continuous-recording CSV and return its per-event loading branches."""
    df = load_continuous_csv(filepath)
    return segment_loading_events(df, sample_name=filepath.stem, **kwargs)
