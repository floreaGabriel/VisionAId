package com.florea_gabriel.impairedhelpapp.presentation.persons

import android.graphics.BitmapFactory
import androidx.compose.animation.*
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
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
import com.florea_gabriel.impairedhelpapp.data.database.KnownPerson
import com.florea_gabriel.impairedhelpapp.presentation.viewmodel.PersonsViewModel
import com.florea_gabriel.impairedhelpapp.ui.theme.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PersonsListScreen(
    viewModel: PersonsViewModel,
    onAddPersonClick: () -> Unit,
    onBackClick: (() -> Unit)? = null,
    bottomPadding: Dp = 0.dp
) {
    val allPersons by viewModel.persons.collectAsState()
    val context = LocalContext.current
    var showDeleteDialog by remember { mutableStateOf<KnownPerson?>(null) }

    Scaffold(
        containerColor = MaterialTheme.colorScheme.background,
        topBar = {
            Surface(
                modifier = Modifier.fillMaxWidth(),
                color = MaterialTheme.colorScheme.surface,
                shadowElevation = AppDimensions.elevationSmall
            ) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(
                            horizontal = AppDimensions.spaceM,
                            vertical = AppDimensions.spaceS
                        ),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    if (onBackClick != null) {
                        IconButton(onClick = onBackClick) {
                            Icon(
                                imageVector = Icons.Default.ArrowBack,
                                contentDescription = "Înapoi",
                                tint = MaterialTheme.colorScheme.onSurface
                            )
                        }
                    }

                    Text(
                        text = "Persoane Cunoscute",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurface,
                        modifier = Modifier.weight(1f)
                    )

                    if (allPersons.isNotEmpty()) {
                        Surface(
                            shape = RoundedCornerShape(AppDimensions.cornerMedium),
                            color = MaterialTheme.colorScheme.primaryContainer
                        ) {
                            Text(
                                text = "${allPersons.size}",
                                style = MaterialTheme.typography.labelMedium,
                                fontWeight = FontWeight.SemiBold,
                                color = MaterialTheme.colorScheme.onPrimaryContainer,
                                modifier = Modifier.padding(horizontal = 12.dp, vertical = 4.dp)
                            )
                        }
                    }
                }
            }
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                onClick = onAddPersonClick,
                containerColor = Indigo,
                contentColor = Color.White,
                modifier = Modifier
                    .padding(AppDimensions.spaceM)
                    .padding(bottom = bottomPadding)
                    .semantics { contentDescription = "Adaugă o persoană nouă" }
            ) {
                Icon(
                    imageVector = Icons.Default.PersonAdd,
                    contentDescription = null,
                    modifier = Modifier.size(AppDimensions.iconMedium)
                )
                Spacer(modifier = Modifier.width(AppDimensions.spaceS))
                Text(text = "Adaugă", style = MaterialTheme.typography.labelLarge)
            }
        }
    ) { padding ->
        Column(modifier = Modifier.fillMaxSize().padding(padding)) {
            when {
                allPersons.isEmpty() -> EmptyState()
                else -> PersonsList(
                    persons = allPersons,
                    onDeleteClick = { showDeleteDialog = it },
                    bottomPadding = bottomPadding
                )
            }
        }
    }

    showDeleteDialog?.let { person ->
        AlertDialog(
            onDismissRequest = { showDeleteDialog = null },
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
                    text = "Șterge persoana?",
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.SemiBold
                )
            },
            text = {
                Text(
                    text = "Sigur vrei să ștergi \"${person.name}\"?\n\nAceastă acțiune nu poate fi anulată.",
                    style = MaterialTheme.typography.bodyLarge,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            },
            confirmButton = {
                Button(
                    onClick = {
                        viewModel.deletePerson(context, person)
                        showDeleteDialog = null
                    },
                    colors = ButtonDefaults.buttonColors(containerColor = Rose),
                    shape = RoundedCornerShape(AppDimensions.cornerSmall)
                ) { Text("Șterge", fontWeight = FontWeight.SemiBold) }
            },
            dismissButton = {
                TextButton(
                    onClick = { showDeleteDialog = null },
                    shape = RoundedCornerShape(AppDimensions.cornerSmall)
                ) { Text("Anulează") }
            },
            shape = RoundedCornerShape(AppDimensions.cornerLarge),
            containerColor = MaterialTheme.colorScheme.surface
        )
    }
}

@Composable
private fun PersonsList(
    persons: List<KnownPerson>,
    onDeleteClick: (KnownPerson) -> Unit,
    bottomPadding: Dp = 0.dp
) {
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(
            horizontal = AppDimensions.spaceM,
            vertical = AppDimensions.spaceS
        ),
        verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceM)
    ) {
        items(items = persons, key = { it.id }) { person ->
            PersonCard(
                person = person,
                onDeleteClick = { onDeleteClick(person) }
            )
        }
        item { Spacer(modifier = Modifier.height(80.dp + bottomPadding)) }
    }
}

@Composable
private fun PersonCard(
    person: KnownPerson,
    onDeleteClick: () -> Unit
) {
    val dateFormat = remember { SimpleDateFormat("dd MMM yyyy", Locale("ro")) }

    Card(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(
                elevation = AppDimensions.elevationMedium,
                shape = RoundedCornerShape(AppDimensions.cornerMedium),
                ambientColor = Color.Black.copy(alpha = 0.05f),
                spotColor = Color.Black.copy(alpha = 0.1f)
            )
            .semantics { contentDescription = "Persoană: ${person.name}" },
        shape = RoundedCornerShape(AppDimensions.cornerMedium),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth().padding(AppDimensions.spaceM),
            horizontalArrangement = Arrangement.spacedBy(AppDimensions.spaceM),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Circular thumbnail
            Box(
                modifier = Modifier
                    .size(64.dp)
                    .clip(CircleShape)
            ) {
                val thumbnailFile = File(person.thumbnailPath)
                if (thumbnailFile.exists()) {
                    var bitmap by remember(person.id) {
                        mutableStateOf(BitmapFactory.decodeFile(person.thumbnailPath))
                    }
                    DisposableEffect(person.id) {
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
                    Box(
                        modifier = Modifier
                            .fillMaxSize()
                            .background(
                                Brush.linearGradient(
                                    colors = listOf(
                                        Indigo.copy(alpha = 0.3f),
                                        Indigo.copy(alpha = 0.1f)
                                    )
                                )
                            ),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Person,
                            contentDescription = null,
                            modifier = Modifier.size(32.dp),
                            tint = Indigo.copy(alpha = 0.5f)
                        )
                    }
                }
            }

            // Name + date
            Column(
                modifier = Modifier.weight(1f),
                verticalArrangement = Arrangement.spacedBy(4.dp)
            ) {
                Text(
                    text = person.name,
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.SemiBold,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                    color = MaterialTheme.colorScheme.onSurface
                )
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
                        text = dateFormat.format(Date(person.createdAt)),
                        style = MaterialTheme.typography.labelSmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                    )
                }
            }

            // Delete button
            IconButton(onClick = onDeleteClick, modifier = Modifier.size(40.dp)) {
                Icon(
                    imageVector = Icons.Default.DeleteOutline,
                    contentDescription = "Șterge ${person.name}",
                    tint = MaterialTheme.colorScheme.error.copy(alpha = 0.7f),
                    modifier = Modifier.size(22.dp)
                )
            }
        }
    }
}

@Composable
private fun EmptyState() {
    Box(
        modifier = Modifier.fillMaxSize().padding(AppDimensions.spaceXL),
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.spacedBy(AppDimensions.spaceL)
        ) {
            Box(
                modifier = Modifier
                    .size(120.dp)
                    .clip(CircleShape)
                    .background(
                        Brush.radialGradient(
                            colors = listOf(
                                Indigo.copy(alpha = 0.15f),
                                Indigo.copy(alpha = 0.05f)
                            )
                        )
                    ),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = Icons.Default.People,
                    contentDescription = null,
                    modifier = Modifier.size(56.dp),
                    tint = Indigo.copy(alpha = 0.6f)
                )
            }

            Text(
                text = "Nu ai persoane salvate",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
                textAlign = TextAlign.Center
            )

            Text(
                text = "Apasă butonul + pentru a\nadăuga prima persoană",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                textAlign = TextAlign.Center
            )
        }
    }
}
