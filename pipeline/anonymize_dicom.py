"""De-identify a DICOM study exported from a patient portal.

Usage: python anonymize_dicom.py <original_export.zip> <anonymized.zip>

What it does (roughly the DICOM PS3.15 "Basic" profile, keeping acquisition data):
  * blanks every person name (patient, physicians, operators)
  * removes patient ID, birth date, address, phone, size/weight, other IDs
  * blanks institution, department, station name and device serial number
  * blanks accession number and study / procedure / step IDs
  * blanks all dates (DA) and date-times (DT); times of day are kept because
    some converters use them for ordering
  * replaces every non-standard UID with a fresh one (consistently within the
    run, so series still group correctly), using a random secret that is not saved
  * removes all private (vendor) tags
  * keeps sex, age, and all scanner / sequence parameters
  * drops the DICOMDIR index and the portal's Readme.html (both contain identifiers)

Image pixels are untouched. Note: a 3D head MRI still shows the face, so the
pixel data itself remains potentially identifying.
"""
import io
import secrets
import sys
import zipfile

import pydicom
from pydicom.uid import generate_uid

REMOVE = {
    "PatientID", "PatientBirthDate", "PatientBirthTime", "OtherPatientIDs", "OtherPatientIDsSequence",
    "OtherPatientNames", "PatientAddress", "PatientTelephoneNumbers", "PatientMotherBirthName",
    "PatientSize", "PatientWeight", "EthnicGroup", "MilitaryRank", "PatientComments",
    "IssuerOfPatientID", "MedicalRecordLocator", "InsurancePlanIdentification",
    "DeviceSerialNumber", "InstitutionAddress", "InstitutionCodeSequence",
}
BLANK = {
    "AccessionNumber", "StudyID", "RequestedProcedureID", "ScheduledProcedureStepID",
    "PerformedProcedureStepID", "InstitutionName", "InstitutionalDepartmentName", "StationName",
    "ImageComments", "AdmissionID", "RequestingService", "PerformedStationName",
    "ScheduledStationName", "PerformedLocation", "ScheduledProcedureStepLocation",
}

salt = secrets.token_hex(16)
uid_map: dict[str, str] = {}


def new_uid(old: str) -> str:
    if old.startswith("1.2.840.10008."):  # standard SOP classes, transfer syntaxes, etc.
        return old
    if old not in uid_map:
        uid_map[old] = generate_uid(entropy_srcs=[salt, old])
    return uid_map[old]


def scrub(ds: pydicom.Dataset) -> None:
    ds.remove_private_tags()
    for el in list(ds.iterall()):
        kw = el.keyword
        if kw in REMOVE:
            el.value = None
        elif kw in BLANK or el.VR in ("PN", "DA", "DT"):
            el.value = "" if el.VR != "SQ" else []
        elif el.VR == "UI" and el.value:
            el.value = [new_uid(u) for u in el.value] if el.VM > 1 else new_uid(str(el.value))
    for kw in REMOVE:
        if kw in ds:
            delattr(ds, kw)
    ds.PatientName = "Anonymous"
    ds.PatientID = "ANON"
    ds.PatientIdentityRemoved = "YES"
    ds.DeidentificationMethod = "anonymize_dicom.py: basic profile, no dates, new UIDs"


def main(src: str, dst: str) -> None:
    n = 0
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zout:
        for info in zin.infolist():
            name = info.filename
            if info.is_dir() or not name.upper().endswith(".DCM"):
                continue  # skips DICOMDIR and Readme.html
            ds = pydicom.dcmread(io.BytesIO(zin.read(name)))
            scrub(ds)
            if ds.file_meta is not None:
                ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
            buf = io.BytesIO()
            ds.save_as(buf, enforce_file_format=True)
            zout.writestr(name, buf.getvalue())
            n += 1
    print(f"anonymized {n} files -> {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
