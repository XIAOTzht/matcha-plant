#!/usr/bin/env python3
"""Read-only dependency and bundle import check; never installs packages."""
from pathlib import Path
import sys
from run_pipeline import check_runtime


def main():
    versions = check_runtime()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'tools/plant_organ_instances'))
    sys.path.insert(0, str(root/'tools/contact_weights'))
    import organ_geometry, split_plant_organs, build_organ_contact_weights, weight_palette, contact_cost
    print('Python:', sys.executable)
    print('Dependencies:', versions)
    print('Palette:', weight_palette.PALETTE_VERSION)
    print('PLANT_CONTACT_ENVIRONMENT_OK (no packages installed)')


if __name__ == '__main__':
    main()
