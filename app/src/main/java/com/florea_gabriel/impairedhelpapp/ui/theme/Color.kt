package com.florea_gabriel.impairedhelpapp.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * VisionAId Color Palette - "Serene Calm"
 *
 * A calming, friendly, and modern color palette designed for accessibility.
 * Inspired by nature (ocean teal, soft greens) to evoke tranquility and trust.
 *
 * Design principles:
 * - Calming teal/cyan as primary (reduces anxiety, builds trust)
 * - Soft, warm accent colors (not aggressive)
 * - High contrast ratios (WCAG AA compliant)
 * - Clear visual hierarchy with gentle transitions
 */

// ============================================
// PRIMARY COLORS - Calming Teal
// ============================================
val Teal = Color(0xFF0D9488) // Primary - calming ocean teal
val TealLight = Color(0xFF14B8A6) // Primary variant (dark mode, lighter)
val TealDark = Color(0xFF0F766E) // Primary pressed state (deeper)
val TealSoft = Color(0xFF5EEAD4) // Very light teal for accents

// Legacy aliases for compatibility
val Indigo = Teal
val IndigoLight = TealLight
val IndigoDark = TealDark

// ============================================
// SEMANTIC COLORS - Softer, friendlier
// ============================================
val Emerald = Color(0xFF22C55E) // Success - warm green
val EmeraldLight = Color(0xFF4ADE80) // Success (dark mode)
val Amber = Color(0xFFFBBF24) // Warning - soft golden
val AmberLight = Color(0xFFFDE68A) // Warning container
val Rose = Color(0xFFF87171) // Error - soft coral (less aggressive)
val RoseLight = Color(0xFFFECACA) // Error container

// ============================================
// LIGHT MODE SURFACES - Warm & inviting
// ============================================
val LightBackground = Color(0xFFF8FAFB) // Subtle warm white
val LightSurface = Color(0xFFFFFFFF) // Pure white - Cards
val LightSurfaceVariant = Color(0xFFF1F5F9) // Soft gray for inputs
val Cream = Color(0xFFFEFCE8) // Warm cream accent

// ============================================
// DARK MODE SURFACES - Comfortable, not harsh
// ============================================
val DarkBackground = Color(0xFF0F1419) // Soft charcoal (warmer than pure black)
val DarkSurface = Color(0xFF1A2332) // Muted slate - Cards
val DarkSurfaceVariant = Color(0xFF2D3A4D) // Lighter slate for inputs

// ============================================
// TEXT COLORS
// ============================================
val TextPrimary = Color(0xFF1E293B) // Slate Dark - Main text
val TextSecondary = Color(0xFF64748B) // Slate Medium - Descriptions
val TextTertiary = Color(0xFF94A3B8) // Slate Light - Hints

val TextPrimaryDark = Color(0xFFF1F5F9) // Light text for dark mode
val TextSecondaryDark = Color(0xFF94A3B8) // Secondary text dark mode

// ============================================
// DIVIDERS & BORDERS
// ============================================
val Divider = Color(0xFFE2E8F0) // Light divider
val DividerDark = Color(0xFF334155) // Dark mode divider

// ============================================
// SPECIAL COLORS - Calming variants
// ============================================
val Scanning = Color(0xFF38BDF8) // Sky blue - softer scanning state
val Found = Color(0xFF22C55E) // Green - Object found
val Listening = Color(0xFFF87171) // Soft coral - Voice listening

// ============================================
// ACCENT COLORS - Gentle touches
// ============================================
val Lavender = Color(0xFFC4B5FD) // Subtle purple accent
val Mint = Color(0xFF99F6E4) // Fresh mint for highlights
val SkyBlue = Color(0xFF7DD3FC) // Light sky for information

// ============================================
// LEGACY COLORS (for compatibility)
// ============================================
val Purple80 = TealLight
val PurpleGrey80 = Color(0xFFB0C4C4)
val Pink80 = Color(0xFFFECACA)

val Purple40 = Teal
val PurpleGrey40 = TextSecondary
val Pink40 = Rose
