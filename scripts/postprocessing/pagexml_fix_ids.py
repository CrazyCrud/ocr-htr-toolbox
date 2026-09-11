import sys
import argparse
import hashlib
from pathlib import Path
from lxml import etree

NS = {"pc": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def get_coords_key(elem):
    coords = elem.find(".//pc:Coords", namespaces=NS)
    if coords is not None:
        return coords.get("points") 
    else:
        return ""


def compute_deterministic_id(base_id, coords_str):
    # Encode coords to bytes
    key = coords_str.encode('utf-8')
    hash_obj = hashlib.sha256(key)
    suffix = hash_obj.hexdigest()[:6]
    return f"{base_id}_{suffix}"

def fix_ids_in_file(path: Path, dry_run: bool = False) -> bool:
    tree = etree.parse(str(path))
    seen_ids = set()
    changed = False

    # Search for TextRegion and TextLine elements
    for elem in tree.xpath('//pc:TextRegion | //pc:TextLine', namespaces=NS):
        element_id = elem.get("id")
        coords_str = get_coords_key(elem)

        base_id = "tr" if elem.tag.endswith("TextRegion") else "tl"
        new_id = compute_deterministic_id(base_id, coords_str)

        if new_id in seen_ids:
            counter = 1
            while f"{new_id}_{counter}" in seen_ids:
                counter += 1
            new_id = f"{new_id}_{counter}"

        if new_id != element_id:
            elem.set("id", new_id)
            print(f"{path.name} ID '{element_id}' to '{new_id}'")
            changed = True

        seen_ids.add(new_id)

    if changed and not dry_run:
        tree.write(str(path), xml_declaration=True, encoding="UTF-8", pretty_print=True)

    return changed

def main():
    p = argparse.ArgumentParser(description="Fix duplicate ids in PAGE-XML using bounding box coords")

    p.add_argument("folder", type=Path,
                   help="Directory to the PAGE-XML files")
    p.add_argument("--dry-run", action="store_true",
                   help="Output changes without editing the actual files")
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
        if fix_ids_in_file(xml, dry_run=args.dry_run):
            total += 1

    print(f"Processed {len(xml_files)} files, fixed duplicates in {total}")
    if args.dry_run:
        print("Dry run, no files were modified")


if __name__ == "__main__":
    main()