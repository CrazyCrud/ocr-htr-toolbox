import sys
import argparse
from pathlib import Path
from lxml import etree

NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def remove_image_path(path: Path, dry_run: bool = False) -> bool:
    tree = etree.parse(str(path))
    changed = False

    # Search for path
    for elem in tree.xpath('//pc:Page', namespaces=NS):
        filename = elem.get("imageFilename")

        image = Path(filename).name

        elem.set("imageFilename", image)

        print(f"{path.name} ID '{filename}' to '{image}'")

        changed = True

    if changed:
        tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)

    return changed


def main():
    p = argparse.ArgumentParser(
        description="Remove folder from image name"
    )
    p.add_argument("folder", type=Path,
                   help="Directory to the PAGE-XML files")

    args = p.parse_args()

    if not args.folder.is_dir():
        print(f"Error: {args.folder} is not a directory")
        sys.exit(1)

    xml_files = args.folder.glob("*.xml")

    if not xml_files:
        print("No .xml files found")
        sys.exit(0)

    total = 0
    for xml in xml_files:
        if remove_image_path(xml):
            total += 1

    print(f"Fixed {total} files")


if __name__ == "__main__":
    main()