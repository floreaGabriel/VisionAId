package com.florea_gabriel.impairedhelpapp.presentation.saved_objects

import android.graphics.BitmapFactory
import androidx.compose.animation.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import com.florea_gabriel.impairedhelpapp.data.database.PersonalObject
import com.florea_gabriel.impairedhelpapp.presentation.viewmodel.SavedObjectsViewModel
import com.florea_gabriel.impairedhelpapp.ui.theme.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

/**
 * SavedObjectsScreen - Modern redesign with card-based layout
 *
 * Design features:
 * - Large touch targets (48dp+ for accessibility)
 * - Soft shadows and rounded corners
 * - Clear visual hierarchy
 * - Generous white space
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SavedObjectsScreen(
        viewModel: SavedObjectsViewModel,
        onBackClick: () -> Unit,
        onAddObjectClick: () -> Unit,
        onSearchClick: (String) -> Unit = {},
        onDetailsClick: (Long) -> Unit = {},
        showBackButton: Boolean = true,
        bottomPadding: Dp = 0.dp
) {
    val allObjects by viewModel.objects.collectAsState()
    val isLoading by viewModel.isLoading.collectAsState()
    val context = LocalContext.current

    var searchQuery by remember { mutableStateOf("") }
    var showDeleteDialog by remember { mutableStateOf<PersonalObject?>(null) }

    val objects =
            remember(allObjects, searchQuery) {
                if (searchQuery.isEmpty()) allObjects
                else
                        allObjects.filter { obj ->
                            obj.name.contains(searchQuery, ignoreCase = true) ||
                                    obj.description.contains(searchQuery, ignoreCase = true)
                        }
            }

    Scaffold(
            containerColor = MaterialTheme.colorScheme.background,
            topBar = {
                // Modern clean top bar with subtle elevation
                Surface(
                        modifier = Modifier.fillMaxWidth(),
                        color = MaterialTheme.colorScheme.surface,
                        shadowElevation = AppDimensions.elevationSmall
                ) {
                    Row(
                            modifier =
                                    Modifier.fillMaxWidth()
                                            .padding(
                                                    horizontal = AppDimensions.spaceM,
                                                    vertical = AppDimensions.spaceS
                                            ),
                            verticalAlignment = Alignment.CenterVertically
                    ) {
                        // Back button (hidden when used as a tab)
                        if (showBackButton) {
                            IconButton(
                                    onClick = onBackClick,
                                    modifier =
                                            Modifier.size(AppDimensions.minTouchTarget).semantics {
                                                contentDescription = "Înapoi la ecranul principal"
                                            }
                            ) {
                                Icon(
                                        imageVector = Icons.Default.ArrowBack,
                                        contentDescription = null,
                                        tint = MaterialTheme.colorScheme.onSurface
                                )
                            }

                            Spacer(modifier = Modifier.width(AppDimensions.spaceS))
                        }

                        // Title
                        Text(
                                text = "Obiectele Mele",
                                style = MaterialTheme.typography.headlineSmall,
                                fontWeight = FontWeight.Bold,
                                color = MaterialTheme.colorScheme.onSurface,
                                modifier = Modifier.weight(1f)
                        )

                        // Object count badge
                        if (allObjects.isNotEmpty()) {
                            Surface(
                                    shape = RoundedCornerShape(AppDimensions.cornerMedium),
                                    color = MaterialTheme.colorScheme.primaryContainer
                            ) {
                                Text(
                                        text = "${allObjects.size}",
                                        style = MaterialTheme.typography.labelMedium,
                                        fontWeight = FontWeight.SemiBold,
                                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                                        modifier =
                                                Modifier.padding(
                                                        horizontal = 12.dp,
                                                        vertical = 4.dp
                                                )
                                )
                            }
                        }
                    }
                }
            },
            floatingActionButton = {}
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            // Search Bar - Modern pill design
            SearchBar(
                    query = searchQuery,
                    onQueryChange = { searchQuery = it },
                    modifier =
                            Modifier.padding(
                                    horizontal = AppDimensions.spaceM,
                                    vertical = AppDimensions.spaceS
                            )
            )

            // Content
            when {
                isLoading -> LoadingState()
                objects.isEmpty() -> EmptyState(isSearching = searchQuery.isNotEmpty())
                else ->
                        ObjectsList(
                                objects = objects,
                                onSearchClick = onSearchClick,
                                onDetailsClick = onDetailsClick,
                                onDeleteClick = { showDeleteDialog = it },
                                bottomPadding = bottomPadding
                        )
            }
        }
    }

    // Delete confirmation dialog
    showDeleteDialog?.let { obj ->
        DeleteConfirmationDialog(
                objectName = obj.name,
                onConfirm = {
                    viewModel.deleteObject(context, obj)
                    showDeleteDialog = null
                },
                onDismiss = { showDeleteDialog = null }
        )
    }
}

@Composable
private fun SearchBar(
        query: String,
        onQueryChange: (String) -> Unit,
        modifier: Modifier = Modifier
) {
    Surface(
            modifier = modifier.fillMaxWidth(),
            shape = RoundedCornerShape(AppDimensions.cornerLarge),
            color = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f)
    ) {
        Row(
                modifier =
                        Modifier.fillMaxWidth()
                                .padding(
                                        horizontal = AppDimensions.spaceM,
                                        vertical = AppDimensions.spaceS
                                ),
                verticalAlignment = Alignment.CenterVertically
        ) {
            Icon(
                    imageVector = Icons.Default.Search,
                    contentDescription = null,
                    tint = MaterialTheme.colorScheme.onSurfaceVariant,
                    modifier = Modifier.size(AppDimensions.iconMedium)
            )

            Spacer(modifier = Modifier.width(AppDimensions.spaceM))

            // Using BasicTextField for more control
            Box(modifier = Modifier.weight(1f)) {
                if (query.isEmpty()) {
                    Text(
                            text = "Caută obiect...",
                            style = MaterialTheme.typography.bodyLarge,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                }
                androidx.compose.foundation.text.BasicTextField(
                        value = query,
                        onValueChange = onQueryChange,
                        singleLine = true,
                        textStyle =
                                MaterialTheme.typography.bodyLarge.copy(
                                        color = MaterialTheme.colorScheme.onSurface
                                ),
                        modifier = Modifier.fillMaxWidth()
                )
            }

            if (query.isNotEmpty()) {
                IconButton(
                        onClick = { onQueryChange("") },
                        modifier = Modifier.size(AppDimensions.minTouchTarget)
                ) {
                    Icon(
                            imageVector = Icons.Default.Close,
                            contentDescription = "Șterge căutarea",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                }
            }
        }
    }
}

@Composable
private fun ObjectsList(
        objects: List<PersonalObject>,
        onSearchClick: (String) -> Unit,
        onDetailsClick: (Long) -> Unit,
        onDeleteClick: (PersonalObject) -> Unit,
        bottomPadding: Dp = 0.dp
) {
    LazyColumn(
            modifier = Modifier.fillMaxSize(),
            contentPadding =
                    PaddingValues(
                            horizontal = AppDimensions.spaceM,
                            vertical = AppDimensions.spaceS
                    ),
            verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
    ) {
        items(items = objects, key = { it.id }) { obj ->
            ModernObjectCard(
                    personalObject = obj,
                    onSearchClick = { onSearchClick(obj.name) },
                    onDetailsClick = { onDetailsClick(obj.id) },
                    onDeleteClick = { onDeleteClick(obj) }
            )
        }

        // Bottom spacer: clears the FAB + HomeScreen bottom nav
        item { Spacer(modifier = Modifier.height(80.dp + bottomPadding)) }
    }
}

@Composable
private fun ModernObjectCard(
        personalObject: PersonalObject,
        onSearchClick: () -> Unit,
        onDetailsClick: () -> Unit,
        onDeleteClick: () -> Unit
) {
    val dateFormat = remember { SimpleDateFormat("dd MMM yyyy", Locale("ro")) }

    Card(
            modifier =
                    Modifier.fillMaxWidth()
                            .shadow(
                                    elevation = AppDimensions.elevationMedium,
                                    shape = RoundedCornerShape(AppDimensions.cornerMedium),
                                    ambientColor = Color.Black.copy(alpha = 0.05f),
                                    spotColor = Color.Black.copy(alpha = 0.1f)
                            )
                            .clickable(onClick = onDetailsClick)
                            .semantics { contentDescription = "Obiect: ${personalObject.name}" },
            shape = RoundedCornerShape(AppDimensions.cornerMedium),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
                modifier = Modifier.fillMaxWidth().padding(AppDimensions.spaceM),
                horizontalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
        ) {
            // Thumbnail with gradient overlay
            Box(
                    modifier =
                            Modifier.size(AppDimensions.thumbnailMedium)
                                    .clip(RoundedCornerShape(AppDimensions.cornerSmall))
            ) {
                val thumbnailFile = File(personalObject.thumbnailPath)
                if (thumbnailFile.exists()) {
                    var bitmap by
                            remember(personalObject.id) {
                                mutableStateOf(
                                        BitmapFactory.decodeFile(personalObject.thumbnailPath)
                                )
                            }

                    DisposableEffect(personalObject.id) {
                        onDispose {
                            bitmap?.recycle()
                            bitmap = null
                        }
                    }

                    bitmap?.let {
                        Image(
                                bitmap = it.asImageBitmap(),
                                contentDescription = null,
                                modifier = Modifier.fillMaxSize(),
                                contentScale = ContentScale.Crop
                        )
                    }
                } else {
                    // Gradient placeholder
                    Box(
                            modifier =
                                    Modifier.fillMaxSize()
                                            .background(
                                                    Brush.linearGradient(
                                                            colors =
                                                                    listOf(
                                                                            Indigo.copy(
                                                                                    alpha = 0.3f
                                                                            ),
                                                                            Indigo.copy(
                                                                                    alpha = 0.1f
                                                                            )
                                                                    )
                                                    )
                                            ),
                            contentAlignment = Alignment.Center
                    ) {
                        Icon(
                                imageVector = Icons.Default.Image,
                                contentDescription = null,
                                modifier = Modifier.size(AppDimensions.iconLarge),
                                tint = Indigo.copy(alpha = 0.5f)
                        )
                    }
                }
            }

            // Content
            Column(
                    modifier = Modifier.weight(1f).height(AppDimensions.thumbnailMedium),
                    verticalArrangement = Arrangement.SpaceBetween
            ) {
                Column {
                    Text(
                            text = personalObject.name,
                            style = MaterialTheme.typography.titleMedium,
                            fontWeight = FontWeight.SemiBold,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            color = MaterialTheme.colorScheme.onSurface
                    )

                    if (personalObject.description.isNotEmpty()) {
                        Text(
                                text = personalObject.description,
                                style = MaterialTheme.typography.bodySmall,
                                maxLines = 2,
                                overflow = TextOverflow.Ellipsis,
                                color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }

                // Date chip
                Row(
                        verticalAlignment = Alignment.CenterVertically,
                        horizontalArrangement = Arrangement.spacedBy(4.dp)
                ) {
                    Icon(
                            imageVector = Icons.Default.Schedule,
                            contentDescription = null,
                            modifier = Modifier.size(14.dp),
                            tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                    )
                    Text(
                            text = dateFormat.format(Date(personalObject.createdAt)),
                            style = MaterialTheme.typography.labelSmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                    )
                }
            }

            // Action buttons - vertical layout
            Column(
                    verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceXS),
                    horizontalAlignment = Alignment.CenterHorizontally
            ) {
                // Search button - Primary action
                FilledIconButton(
                        onClick = onSearchClick,
                        modifier = Modifier.size(AppDimensions.minTouchTarget),
                        colors =
                                IconButtonDefaults.filledIconButtonColors(
                                        containerColor = Emerald,
                                        contentColor = Color.White
                                )
                ) {
                    Icon(
                            imageVector = Icons.Default.Search,
                            contentDescription = "Caută ${personalObject.name}",
                            modifier = Modifier.size(AppDimensions.iconMedium)
                    )
                }

                // Delete button
                IconButton(onClick = onDeleteClick, modifier = Modifier.size(36.dp)) {
                    Icon(
                            imageVector = Icons.Default.DeleteOutline,
                            contentDescription = "Șterge",
                            tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                            modifier = Modifier.size(20.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun LoadingState() {
    Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
        ) {
            CircularProgressIndicator(color = Indigo, strokeWidth = 3.dp)
            Text(
                    text = "Se încarcă...",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
            )
        }
    }
}

@Composable
private fun EmptyState(isSearching: Boolean) {
    Box(
            modifier = Modifier.fillMaxSize().padding(AppDimensions.spaceXL),
            contentAlignment = Alignment.Center
    ) {
        Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceL)
        ) {
            // Icon with gradient background
            Box(
                    modifier =
                            Modifier.size(120.dp)
                                    .clip(CircleShape)
                                    .background(
                                            Brush.radialGradient(
                                                    colors =
                                                            listOf(
                                                                    Indigo.copy(alpha = 0.15f),
                                                                    Indigo.copy(alpha = 0.05f)
                                                            )
                                            )
                                    ),
                    contentAlignment = Alignment.Center
            ) {
                Icon(
                        imageVector =
                                if (isSearching) Icons.Default.SearchOff
                                else Icons.Default.Inventory2,
                        contentDescription = null,
                        modifier = Modifier.size(56.dp),
                        tint = Indigo.copy(alpha = 0.6f)
                )
            }

            Text(
                    text = if (isSearching) "Niciun rezultat" else "Niciun obiect salvat",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onSurface,
                    textAlign = TextAlign.Center
            )

            Text(
                    text =
                            if (isSearching) "Încearcă să cauți altceva"
                            else "Apasă butonul + pentru a\nadăuga primul tău obiect",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                    textAlign = TextAlign.Center
            )
        }
    }
}

@Composable
private fun DeleteConfirmationDialog(
        objectName: String,
        onConfirm: () -> Unit,
        onDismiss: () -> Unit
) {
    AlertDialog(
            onDismissRequest = onDismiss,
            icon = {
                Icon(
                        imageVector = Icons.Default.Warning,
                        contentDescription = null,
                        tint = Rose,
                        modifier = Modifier.size(AppDimensions.iconLarge)
                )
            },
            title = {
                Text(
                        text = "Șterge obiectul?",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Text(
                        text =
                                "Sigur vrei să ștergi \"$objectName\"?\n\nAceastă acțiune nu poate fi anulată.",
                        style = MaterialTheme.typography.bodyLarge,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            },
            confirmButton = {
                Button(
                        onClick = onConfirm,
                        colors = ButtonDefaults.buttonColors(containerColor = Rose),
                        shape = RoundedCornerShape(AppDimensions.cornerSmall)
                ) { Text("Șterge", fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(
                        onClick = onDismiss,
                        shape = RoundedCornerShape(AppDimensions.cornerSmall)
                ) { Text("Anulează") }
            },
            shape = RoundedCornerShape(AppDimensions.cornerLarge),
            containerColor = MaterialTheme.colorScheme.surface
    )
}
