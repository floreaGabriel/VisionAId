package com.florea_gabriel.impairedhelpapp.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/**
 * VisionAId Theme - "Serene Calm"
 *
 * A calming, friendly, and modern theme with:
 * - Teal/cyan primary colors (evokes trust and tranquility)
 * - Soft, warm accent colors
 * - Light and Dark mode support
 * - Accessibility-focused design
 * - Gentle visual hierarchy
 */

// Additional colors for theme containers (inline definitions)
private val SecondaryContainerLight = Color(0xFFD1FAE5) // Soft green 100
private val OnSecondaryContainerLight = Color(0xFF065F46) // Green 800
private val TertiaryContainerLight = Color(0xFFFEF3C7) // Soft amber 100
private val OnTertiaryContainerLight = Color(0xFF92400E) // Amber 800
private val ErrorContainerLight = Color(0xFFFEE2E2) // Soft rose 100
private val OnErrorContainerLight = Color(0xFF991B1B) // Rose 800
private val OutlineVariantLight = Color(0xFFCBD5E1) // Slate 300

private val ErrorDark = Color(0xFFFCA5A5) // Soft coral 300
private val OnErrorDark = Color(0xFF7F1D1D)
private val ErrorContainerDark = Color(0xFF991B1B)
private val OnErrorContainerDark = Color(0xFFFEE2E2)
private val OutlineVariantDark = Color(0xFF475569) // Slate 600

private val LightColorScheme =
        lightColorScheme(
                // Primary
                primary = Indigo,
                onPrimary = LightSurface,
                primaryContainer = IndigoLight,
                onPrimaryContainer = TextPrimary,

                // Secondary
                secondary = Emerald,
                onSecondary = LightSurface,
                secondaryContainer = SecondaryContainerLight,
                onSecondaryContainer = OnSecondaryContainerLight,

                // Tertiary (Warning/Accent)
                tertiary = Amber,
                onTertiary = TextPrimary,
                tertiaryContainer = TertiaryContainerLight,
                onTertiaryContainer = OnTertiaryContainerLight,

                // Error
                error = Rose,
                onError = LightSurface,
                errorContainer = ErrorContainerLight,
                onErrorContainer = OnErrorContainerLight,

                // Background & Surface
                background = LightBackground,
                onBackground = TextPrimary,
                surface = LightSurface,
                onSurface = TextPrimary,
                surfaceVariant = LightSurfaceVariant,
                onSurfaceVariant = TextSecondary,

                // Outline
                outline = Divider,
                outlineVariant = OutlineVariantLight,

                // Inverse (for snackbars, etc.)
                inverseSurface = DarkSurface,
                inverseOnSurface = TextPrimaryDark,
                inversePrimary = IndigoLight,

                // Other
                scrim = Color.Black
        )

private val DarkColorScheme =
        darkColorScheme(
                // Primary
                primary = IndigoLight,
                onPrimary = DarkBackground,
                primaryContainer = Indigo,
                onPrimaryContainer = TextPrimaryDark,

                // Secondary
                secondary = EmeraldLight,
                onSecondary = DarkBackground,
                secondaryContainer = OnSecondaryContainerLight,
                onSecondaryContainer = SecondaryContainerLight,

                // Tertiary
                tertiary = Amber,
                onTertiary = DarkBackground,
                tertiaryContainer = OnTertiaryContainerLight,
                onTertiaryContainer = TertiaryContainerLight,

                // Error
                error = ErrorDark,
                onError = OnErrorDark,
                errorContainer = ErrorContainerDark,
                onErrorContainer = OnErrorContainerDark,

                // Background & Surface
                background = DarkBackground,
                onBackground = TextPrimaryDark,
                surface = DarkSurface,
                onSurface = TextPrimaryDark,
                surfaceVariant = DarkSurfaceVariant,
                onSurfaceVariant = TextSecondaryDark,

                // Outline
                outline = DividerDark,
                outlineVariant = OutlineVariantDark,

                // Inverse
                inverseSurface = LightSurface,
                inverseOnSurface = TextPrimary,
                inversePrimary = Indigo,

                // Other
                scrim = Color.Black
        )

@Composable
fun VisionAIdTheme(
        darkTheme: Boolean = isSystemInDarkTheme(),
        content: @Composable () -> Unit
) {
        val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

        // System bar colors are now handled in themes.xml
        // This ensures the app respects status bar and navigation bar

        MaterialTheme(colorScheme = colorScheme, typography = Typography, content = content)
}
