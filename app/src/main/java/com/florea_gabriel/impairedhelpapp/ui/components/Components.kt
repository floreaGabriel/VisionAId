package com.florea_gabriel.impairedhelpapp.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import com.florea_gabriel.impairedhelpapp.ui.theme.*

/**
 * AppButton - Primary button component
 *
 * Variants:
 * - Primary: Filled with primary color
 * - Secondary: Outlined with primary color
 * - Success: Filled with success/emerald color
 * - Ghost: Text-only button
 */
enum class ButtonVariant {
        Primary,
        Secondary,
        Success,
        Ghost
}

@Composable
fun AppButton(
        text: String,
        onClick: () -> Unit,
        modifier: Modifier = Modifier,
        variant: ButtonVariant = ButtonVariant.Primary,
        icon: ImageVector? = null,
        enabled: Boolean = true,
        fullWidth: Boolean = false,
        contentDescription: String? = null
) {
        val buttonModifier =
                modifier.height(AppDimensions.buttonHeightMedium)
                        .then(if (fullWidth) Modifier.fillMaxWidth() else Modifier)
                        .semantics {
                                this.role = Role.Button
                                contentDescription?.let { this.contentDescription = it }
                        }

        val shape = RoundedCornerShape(AppDimensions.cornerMedium)

        when (variant) {
                ButtonVariant.Primary -> {
                        Button(
                                onClick = onClick,
                                modifier = buttonModifier,
                                enabled = enabled,
                                shape = shape,
                                colors =
                                        ButtonDefaults.buttonColors(
                                                containerColor = Indigo,
                                                contentColor = Color.White,
                                                disabledContainerColor = Indigo.copy(alpha = 0.5f),
                                                disabledContentColor =
                                                        Color.White.copy(alpha = 0.5f)
                                        ),
                                elevation =
                                        ButtonDefaults.buttonElevation(
                                                defaultElevation = AppDimensions.elevationSmall,
                                                pressedElevation = AppDimensions.elevationMedium
                                        )
                        ) { ButtonContent(icon = icon, text = text) }
                }
                ButtonVariant.Secondary -> {
                        OutlinedButton(
                                onClick = onClick,
                                modifier = buttonModifier,
                                enabled = enabled,
                                shape = shape,
                                border =
                                        BorderStroke(
                                                1.5.dp,
                                                if (enabled) Indigo else Indigo.copy(alpha = 0.5f)
                                        ),
                                colors =
                                        ButtonDefaults.outlinedButtonColors(
                                                contentColor = Indigo,
                                                disabledContentColor = Indigo.copy(alpha = 0.5f)
                                        )
                        ) { ButtonContent(icon = icon, text = text, tint = Indigo) }
                }
                ButtonVariant.Success -> {
                        Button(
                                onClick = onClick,
                                modifier = buttonModifier,
                                enabled = enabled,
                                shape = shape,
                                colors =
                                        ButtonDefaults.buttonColors(
                                                containerColor = Emerald,
                                                contentColor = Color.White,
                                                disabledContainerColor = Emerald.copy(alpha = 0.5f),
                                                disabledContentColor =
                                                        Color.White.copy(alpha = 0.5f)
                                        ),
                                elevation =
                                        ButtonDefaults.buttonElevation(
                                                defaultElevation = AppDimensions.elevationSmall,
                                                pressedElevation = AppDimensions.elevationMedium
                                        )
                        ) { ButtonContent(icon = icon, text = text) }
                }
                ButtonVariant.Ghost -> {
                        TextButton(
                                onClick = onClick,
                                modifier = buttonModifier,
                                enabled = enabled,
                                shape = shape,
                                colors =
                                        ButtonDefaults.textButtonColors(
                                                contentColor = Indigo,
                                                disabledContentColor = Indigo.copy(alpha = 0.5f)
                                        )
                        ) { ButtonContent(icon = icon, text = text, tint = Indigo) }
                }
        }
}

@Composable
private fun ButtonContent(icon: ImageVector?, text: String, tint: Color = Color.White) {
        if (icon != null) {
                Icon(
                        imageVector = icon,
                        contentDescription = null,
                        modifier = Modifier.size(AppDimensions.iconMedium),
                        tint = tint
                )
                Spacer(modifier = Modifier.width(AppDimensions.spaceS))
        }
        Text(text = text, style = MaterialTheme.typography.labelLarge)
}

/** AppCard - Elevated card component with consistent styling */
@Composable
fun AppCard(
        modifier: Modifier = Modifier,
        onClick: (() -> Unit)? = null,
        contentPadding: PaddingValues = PaddingValues(AppDimensions.spaceM),
        content: @Composable ColumnScope.() -> Unit
) {
        val cardModifier = modifier.fillMaxWidth()
        val shape = RoundedCornerShape(AppDimensions.cornerMedium)
        val colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
        val elevation =
                CardDefaults.cardElevation(
                        defaultElevation = AppDimensions.elevationSmall,
                        pressedElevation = AppDimensions.elevationMedium
                )

        if (onClick != null) {
                Card(
                        onClick = onClick,
                        modifier = cardModifier,
                        shape = shape,
                        colors = colors,
                        elevation = elevation
                ) { Column(modifier = Modifier.padding(contentPadding), content = content) }
        } else {
                Card(
                        modifier = cardModifier,
                        shape = shape,
                        colors = colors,
                        elevation = elevation
                ) { Column(modifier = Modifier.padding(contentPadding), content = content) }
        }
}

/** AppTopBar - Consistent top app bar */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppTopBar(
        title: String,
        modifier: Modifier = Modifier,
        onBackClick: (() -> Unit)? = null,
        actions: @Composable RowScope.() -> Unit = {}
) {
        TopAppBar(
                title = { Text(text = title, style = MaterialTheme.typography.titleLarge) },
                modifier = modifier,
                navigationIcon = {
                        if (onBackClick != null) {
                                IconButton(
                                        onClick = onBackClick,
                                        modifier = Modifier.size(AppDimensions.minTouchTarget)
                                ) {
                                        Icon(
                                                imageVector = Icons.Default.ArrowBack,
                                                contentDescription = "Înapoi",
                                                modifier = Modifier.size(AppDimensions.iconMedium)
                                        )
                                }
                        }
                },
                actions = actions,
                colors =
                        TopAppBarDefaults.topAppBarColors(
                                containerColor = MaterialTheme.colorScheme.surface,
                                titleContentColor = MaterialTheme.colorScheme.onSurface,
                                navigationIconContentColor = MaterialTheme.colorScheme.onSurface,
                                actionIconContentColor = MaterialTheme.colorScheme.onSurfaceVariant
                        )
        )
}

/** AppEmptyState - Empty state placeholder */
@Composable
fun AppEmptyState(
        icon: ImageVector,
        title: String,
        description: String,
        modifier: Modifier = Modifier,
        action: (@Composable () -> Unit)? = null
) {
        Column(
                modifier = modifier.fillMaxWidth().padding(AppDimensions.spaceXL),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
        ) {
                Icon(
                        imageVector = icon,
                        contentDescription = null,
                        modifier = Modifier.size(AppDimensions.iconXLarge),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )

                Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.onSurface,
                        textAlign = TextAlign.Center
                )

                Text(
                        text = description,
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        textAlign = TextAlign.Center
                )

                if (action != null) {
                        Spacer(modifier = Modifier.height(AppDimensions.spaceS))
                        action()
                }
        }
}

/** AppDivider - Consistent divider */
@Composable
fun AppDivider(modifier: Modifier = Modifier) {
        HorizontalDivider(
                modifier = modifier.padding(vertical = AppDimensions.spaceS),
                thickness = AppDimensions.dividerThickness,
                color = MaterialTheme.colorScheme.outlineVariant
        )
}
