# Retrospective growth records: what to collect (DRAFT)

Purpose: a second real longitudinal growth dataset to validate the growth twin (M11). One row
per measurement, in `data/schema/growth_records_template.csv` format. The example rows are
invented and only show the format.

## Before anything is copied (non-negotiable)
- Written permission from the school or academy that holds the records.
- Ethics approval or exemption for a retrospective study of de-identified records.
  TODO(human): obtain both; record the reference numbers here.
- Names never enter the file. Each child gets a made-up `child_code`; the code-to-name list
  stays with the institution, outside this repository.
- No photographs or video.

## Fields
| Column | Required | Format |
|---|---|---|
| `child_code` | yes | made-up ID, e.g. `S001` |
| `sex` | yes | `male` / `female` |
| `birth_month` | yes | `YYYY-MM` (exact day not needed) |
| `measurement_date` | yes | `YYYY-MM-DD` |
| `stature_cm` | yes | standing height, 0.1 cm |
| `sitting_height_cm` | strongly wanted | 0.1 cm; needed for Moore (boys) |
| `mass_kg` | wanted | 0.1 kg; needed for Mirwald |
| `mother_height_cm`, `father_height_cm` | optional | |
| `setting` | optional | school or academy label, no names |

## Usable data
- Ages about 8-18 at measurement.
- At least three measurements per child spanning at least two years (more is better).
- At least 100 such children; 200 or more is much stronger. Both sexes if possible.
- Note how heights were measured (stadiometer or wall chart, shoes off) and by whom.

## Where such records usually exist
Annual school medical check-ups, PE or fitness-test registers, academy screening sheets,
sports-hostel health cards.
