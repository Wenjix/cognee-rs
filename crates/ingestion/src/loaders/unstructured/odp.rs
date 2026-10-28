//! ODP (OpenDocument Presentation) text extraction via `zip` + `quick-xml`.
//!
//! Opens the ODP file as a ZIP archive, reads `content.xml`, and
//! extracts text from `<text:p>` elements (same parser as ODT).
//! Paragraphs are joined with `"\n\n"`.

use std::io::Cursor;

use super::super::LoaderError;

/// Extract text from an ODP file.
pub(crate) fn extract(bytes: &[u8]) -> Result<String, LoaderError> {
    extract_with_limit(bytes, super::super::MAX_EXTRACTED_BYTES)
}

fn extract_with_limit(bytes: &[u8], limit: usize) -> Result<String, LoaderError> {
    let cursor = Cursor::new(bytes);
    let mut archive = zip::ZipArchive::new(cursor)
        .map_err(|e| LoaderError::ExtractionFailed(format!("Failed to open ODP as ZIP: {e}")))?;

    let mut budget = limit;
    let xml_content = {
        let file = archive.by_name("content.xml").map_err(|e| {
            LoaderError::ExtractionFailed(format!("Failed to find content.xml in ODP: {e}"))
        })?;
        super::read_entry_to_string(file, "content.xml", &mut budget)?
    };

    // Reuse the ODF content parser from the ODT module
    super::odt::parse_odf_content(&xml_content)
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test code — panics are acceptable failures"
)]
mod tests {
    use super::*;

    #[test]
    fn invalid_zip_returns_error() {
        let result = extract(b"not a zip file");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, LoaderError::ExtractionFailed(_)),
            "expected ExtractionFailed, got {err:?}"
        );
    }

    #[test]
    fn empty_bytes_returns_error() {
        let result = extract(b"");
        assert!(result.is_err());
    }

    #[test]
    fn a_decompressed_entry_past_the_limit_is_rejected() {
        let big = vec![b' '; 2_000_000];
        let zip = super::super::zip_bytes(&[("content.xml", big.as_slice())]);
        assert!(matches!(
            extract_with_limit(&zip, 1_000_000),
            Err(LoaderError::ExtractionFailed(_))
        ));
    }

    #[test]
    fn a_small_document_extracts_exactly_as_before() {
        let xml = br#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:presentation><text:p>hi there</text:p></office:presentation></office:body></office:document-content>"#;
        let zip = super::super::zip_bytes(&[("content.xml", xml)]);
        let text = extract_with_limit(&zip, 1_000_000).unwrap();
        assert!(text.contains("hi there"), "{text:?}");
    }
}
