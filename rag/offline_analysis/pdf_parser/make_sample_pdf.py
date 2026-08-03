from pathlib import Path

import pymupdf


def main() -> None:
    output = Path("sample.pdf")
    doc = pymupdf.open()

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 45), "RAG PDF Parser Demo", fontsize=9)
    page.insert_text((72, 800), "Page 1", fontsize=9)
    page.insert_text((72, 100), "1. User Management", fontsize=20)
    page.insert_text(
        (72, 145),
        "Administrators can create users and reset passwords.",
        fontsize=12,
    )
    page.insert_text(
        (72, 175),
        "Open Settings, select Security, enter a new password, and save.",
        fontsize=12,
    )

    page = doc.new_page(width=595, height=842)
    page.insert_text((72, 45), "RAG PDF Parser Demo", fontsize=9)
    page.insert_text((72, 800), "Page 2", fontsize=9)
    page.insert_text((72, 100), "2. Permission Management", fontsize=20)
    page.insert_text(
        (72, 145),
        "Roles group permissions. Users obtain permissions through roles.",
        fontsize=12,
    )

    doc.save(output)
    doc.close()
    print(output.resolve())


if __name__ == "__main__":
    main()
