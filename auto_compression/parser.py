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
# Continuous-recording format (e.g. "Robustness Test ... (Raw).csv")
#
# Unlike the per-specimen Instron exports above (one ``SID,"..."`` block with
# repeated ``N,Time,Displacement,...`` section headers), these files are a
# single uninterrupted recording in which many compression events run
# back-to-back. There are no section-header rows to split on, the strain column
# is ``Compressive strain (Displacement)`` and the column order is
# strain-before-stress (the opposite of ``usecols=[4, 5]`` above), so the
# section-based parser cannot read them. Segment by detecting strain peaks.
# ---------------------------------------------------------------------------

STRAIN_COL = 'Compressive strain (Displacement)'  # X (%)
STRESS_COL = 'Compressive stress'                 # Y (MPa)


def load_continuous_csv(filepath: Path) -> pd.DataFrame:
    """Load a continuous multi-event recording.

    Row 0 is the column header, row 1 is the units row (dropped). All columns
    are coerced to numeric and rows with missing strain/stress are removed.
    """
    df = pd.read_csv(filepath, skiprows=[1])
    df = df.apply(pd.to_numeric, errors='coerce')
    return df.dropna(subset=[STRAIN_COL, STRESS_COL]).reset_index(drop=True)


def segment_loading_events(
    df: pd.DataFrame,
    sample_name: str = 'continuous',
    *,
    peak_height: float = 30.0,
    peak_distance: int = 100,
    peak_prominence: float = 20.0,
    liftoff_stress: float = 0.005,
) -> list[SpecimenData]:
    """Split a continuous recording into per-event loading branches.

    Each compression event ramps strain up to a peak then unloads. We locate the
    strain peaks, walk backward from each peak to the lift-off point (last sample
    before stress rises above ``liftoff_stress``), and return the loading branch
    (lift-off -> peak) with strain re-zeroed at lift-off, so every event starts
    at ``(0, ~0)``. The unloading branch is intentionally discarded.

    ``specimen_num`` is the 1-based event index in acquisition order.
    """
    strain = df[STRAIN_COL].values
    stress = df[STRESS_COL].values
    peaks, _ = find_peaks(strain, height=peak_height,
                          distance=peak_distance, prominence=peak_prominence)

    events = []
    for event_num, p in enumerate(peaks, 1):
        i = p
        while i > 0 and stress[i] >= liftoff_stress:
            i -= 1
        i_start = i + 1
        e0 = float(strain[i_start])
        rel_strain = strain[i_start:p + 1] - e0
        events.append(SpecimenData(
            sample_name=sample_name,
            specimen_num=event_num,
            strain=pd.Series(rel_strain).reset_index(drop=True),
            stress=pd.Series(stress[i_start:p + 1]).reset_index(drop=True),
        ))
    return events


def parse_continuous_csv(filepath: Path, **kwargs) -> list[SpecimenData]:
    """Load a continuous-recording CSV and return its per-event loading branches."""
    df = load_continuous_csv(filepath)
    return segment_loading_events(df, sample_name=filepath.stem, **kwargs)
