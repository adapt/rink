//! Still-image preprocessing for the released Inkling processor. Preserve pixels,
//! composite transparency on its checkerboard, and pad the right and bottom edges.
use anyhow::{Context, Result, ensure};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use std::path::Path;

pub const IMAGE_TOKEN: &str = "<|unused_200054|>";
pub const MAX_IMAGE_DIMENSION: u32 = 16_384;

#[derive(Clone, Debug)]
pub struct Image {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

impl Image {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut reader = ImageReader::open(path)
            .with_context(|| format!("opening image {}", path.display()))?
            .with_guessed_format()?;
        ensure!(
            matches!(reader.format(), Some(ImageFormat::Png | ImageFormat::Jpeg)),
            "image must be a PNG or JPEG file"
        );
        let mut limits = Limits::default();
        limits.max_image_width = Some(MAX_IMAGE_DIMENSION);
        limits.max_image_height = Some(MAX_IMAGE_DIMENSION);
        // Allow a full-size RGBA image plus decoder working memory. The default
        // 512 MiB budget is smaller than a 16K image's decoded pixels.
        limits.max_alloc = Some(u64::from(MAX_IMAGE_DIMENSION).pow(2) * 8);
        reader.limits(limits);
        let mut decoder = reader.into_decoder().with_context(|| {
            format!("decoding image (maximum dimension is {MAX_IMAGE_DIMENSION} pixels)")
        })?;
        let color = decoder.color_type();
        ensure!(
            color.bits_per_pixel() == u16::from(color.channel_count()) * 8,
            "image must use 8-bit channels"
        );
        let orientation = decoder.orientation()?;
        let mut decoded = DynamicImage::from_decoder(decoder)?;
        decoded.apply_orientation(orientation);
        let rgba = decoded.into_rgba8();
        Self::from_rgba(rgba.width(), rgba.height(), rgba.into_raw())
    }

    pub fn from_rgba(width: u32, height: u32, rgba: Vec<u8>) -> Result<Self> {
        ensure!(
            (1..=MAX_IMAGE_DIMENSION).contains(&width)
                && (1..=MAX_IMAGE_DIMENSION).contains(&height),
            "image dimensions must be in 1..={MAX_IMAGE_DIMENSION}"
        );
        ensure!(
            rgba.len() == width as usize * height as usize * 4,
            "invalid RGBA buffer length"
        );
        Ok(Self {
            width: width as usize,
            height: height as usize,
            rgba,
        })
    }
    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }
    pub fn grid(&self) -> (usize, usize) {
        (self.height.div_ceil(40), self.width.div_ceil(40))
    }
    pub fn token_count(&self) -> usize {
        let (rows, cols) = self.grid();
        rows * cols
    }

    /// Native 20x20 RGB patches, grouped in 2x2 blocks in image-token order.
    /// The reference fuses rescale/normalize into (pixel - 127.5) / 127.5.
    pub fn patches(&self) -> Vec<f32> {
        let (rows, cols) = self.grid();
        let mut out = Vec::with_capacity(rows * cols * 40 * 40 * 3);
        for ty in 0..rows {
            for tx in 0..cols {
                for sy in 0..2 {
                    for sx in 0..2 {
                        for py in 0..20 {
                            for px in 0..20 {
                                let y = ty * 40 + sy * 20 + py;
                                let x = tx * 40 + sx * 20 + px;
                                for c in 0..3 {
                                    let pixel = if x < self.width && y < self.height {
                                        let i = (y * self.width + x) * 4;
                                        let alpha = u32::from(self.rgba[i + 3]);
                                        let background =
                                            if (x / 8 + y / 8) % 2 == 0 { 204 } else { 153 };
                                        ((u32::from(self.rgba[i + c]) * alpha
                                            + background * (255 - alpha)
                                            + 127)
                                            / 255) as f32
                                    } else {
                                        -1.
                                    };
                                    out.push((pixel - 127.5) / 127.5);
                                }
                            }
                        }
                    }
                }
            }
        }
        out
    }
}

/// Image part followed by the user's question, as in the released chat template.
pub fn image_user_message(prompt: &str, tokens: usize) -> String {
    format!(
        "<|message_user|><|content_image|>{}<|end_message|>{}",
        IMAGE_TOKEN.repeat(tokens),
        crate::chat_user_message(prompt)
    )
}

pub fn image_prompt(prompt: &str, tokens: usize, effort: f32) -> Result<String> {
    // Reuse the text template's validated system prefix, replacing only the turn.
    let prefix = crate::chat_prompt("", effort)?;
    let suffix = crate::chat_user_message("");
    Ok(format!(
        "{}{}",
        prefix.strip_suffix(&suffix).unwrap(),
        image_user_message(prompt, tokens)
    ))
}
