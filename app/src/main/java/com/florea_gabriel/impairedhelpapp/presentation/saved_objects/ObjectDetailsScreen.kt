package com.florea_gabriel.impairedhelpapp.presentation.saved_objects

import android.graphics.BitmapFactory
import android.os.Environment
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.florea_gabriel.impairedhelpapp.data.database.PersonalObject
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import org.json.JSONArray

/**
 * ObjectDetailsScreen: Debug screen showing all saved data about a registered object.
 *
 * Displays:
 * - Object metadata (ID, name, description, timestamps)
 * - Gemini AI metadata (label, description, keywords)
 * - Embeddings info (count, size, preview)
 * - Debug images from segmentation (if available)
 *
 * @param personalObject The object to display details for
 * @param onBackClick Callback when back button is pressed
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ObjectDetailsScreen(personalObject: PersonalObject, onBackClick: () -> Unit) {
    val context = LocalContext.current
    val dateFormat = remember { SimpleDateFormat("dd MMM yyyy, HH:mm:ss", Locale("ro")) }
    val scrollState = rememberScrollState()

    // Parse embeddings info
    val embeddingsInfo =
            remember(personalObject.embeddingJson) {
                parseEmbeddingsInfo(personalObject.embeddingJson)
            }

    // Find debug images
    val debugImages = remember { findDebugImages(context) }

    Scaffold(
            topBar = {
                TopAppBar(
                        title = { Text("Detalii Debug", fontWeight = FontWeight.Bold) },
                        navigationIcon = {
                            IconButton(onClick = onBackClick) {
                                Icon(
                                        imageVector = Icons.Default.ArrowBack,
                                        contentDescription = "Înapoi"
                                )
                            }
                        },
                        colors =
                                TopAppBarDefaults.topAppBarColors(
                                        containerColor = MaterialTheme.colorScheme.primaryContainer,
                                        titleContentColor =
                                                MaterialTheme.colorScheme.onPrimaryContainer
                                )
                )
            }
    ) { padding ->
        Column(
                modifier =
                        Modifier.fillMaxSize()
                                .padding(padding)
                                .verticalScroll(scrollState)
                                .padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header with thumbnail and name
            HeaderSection(personalObject)

            HorizontalDivider()

            // General Info Section
            InfoSection(title = "📋 Informații Generale", icon = Icons.Default.Info) {
                InfoRow("ID", personalObject.id.toString())
                InfoRow("Nume", personalObject.name)
                if (personalObject.description.isNotEmpty()) {
                    InfoRow("Descriere", personalObject.description)
                }
                InfoRow("Creat la", dateFormat.format(Date(personalObject.createdAt)))
                if (personalObject.lastSearchedAt > 0) {
                    InfoRow(
                            "Ultima căutare",
                            dateFormat.format(Date(personalObject.lastSearchedAt))
                    )
                } else {
                    InfoRow("Ultima căutare", "Niciodată")
                }
            }

            // Gemini AI Metadata Section (if available)
            if (personalObject.geminiLabel != null ||
                            personalObject.geminiDescription != null ||
                            personalObject.detectionKeywords != null
            ) {

                InfoSection(title = "🤖 Gemini AI Metadata", icon = Icons.Default.Psychology) {
                    personalObject.geminiLabel?.let { InfoRow("Label", it) }
                    personalObject.geminiDescription?.let { InfoRow("Descriere AI", it) }
                    personalObject.detectionKeywords?.let { InfoRow("Keywords YOLO", it) }
                }
            }

            // Embeddings Section
            InfoSection(title = "🧠 CLIP Embeddings", icon = Icons.Default.Memory) {
                InfoRow("Număr embeddings", embeddingsInfo.count.toString())
                InfoRow("Dimensiune embedding", "${embeddingsInfo.dimension} valori")
                InfoRow("JSON size", "${embeddingsInfo.jsonSize} bytes")

                // Adaptive threshold
                val threshold = personalObject.recommendedThreshold
                if (threshold != null) {
                    InfoRow("🎯 Threshold căutare", String.format("%.2f", threshold))
                } else {
                    InfoRow("🎯 Threshold căutare", "0.60 (default)")
                }

                if (embeddingsInfo.previewValues.isNotEmpty()) {
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                            text = "Preview (primele 5 valori):",
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = FontWeight.Medium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                    )
                    Text(
                            text = embeddingsInfo.previewValues,
                            style = MaterialTheme.typography.bodySmall,
                            fontFamily = FontFamily.Monospace,
                            fontSize = 10.sp,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                            modifier =
                                    Modifier.fillMaxWidth()
                                            .background(
                                                    MaterialTheme.colorScheme.surfaceVariant,
                                                    RoundedCornerShape(4.dp)
                                            )
                                            .padding(8.dp)
                    )
                }
            }

            // Thumbnail Path Section
            InfoSection(title = "📁 Fișiere", icon = Icons.Default.Folder) {
                Text(
                        text = "Thumbnail path:",
                        style = MaterialTheme.typography.bodySmall,
                        fontWeight = FontWeight.Medium
                )
                Text(
                        text = personalObject.thumbnailPath,
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        fontSize = 10.sp,
                        color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                        maxLines = 2,
                        overflow = TextOverflow.Ellipsis
                )
            }

            // Debug Images Section (if available)
            if (debugImages.isNotEmpty()) {
                InfoSection(
                        title = "📸 Imagini Debug Segmentare (${debugImages.size})",
                        icon = Icons.Default.PhotoLibrary
                ) { DebugImagesGallery(debugImages) }
            } else {
                InfoSection(
                        title = "📸 Imagini Debug Segmentare",
                        icon = Icons.Default.PhotoLibrary
                ) {
                    Text(
                            text =
                                    "Nu există imagini de debug.\n\nImagini se salvează în timpul înregistrării unui obiect nou.",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                    )
                }
            }

            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}

@Composable
private fun HeaderSection(personalObject: PersonalObject) {
    Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(16.dp),
            verticalAlignment = Alignment.CenterVertically
    ) {
        // Thumbnail
        val thumbnailFile = File(personalObject.thumbnailPath)
        if (thumbnailFile.exists()) {
            var bitmap by
                    remember(personalObject.id) {
                        mutableStateOf(BitmapFactory.decodeFile(personalObject.thumbnailPath))
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
                        contentDescription = personalObject.name,
                        modifier =
                                Modifier.size(100.dp)
                                        .clip(RoundedCornerShape(12.dp))
                                        .border(
                                                2.dp,
                                                MaterialTheme.colorScheme.primary,
                                                RoundedCornerShape(12.dp)
                                        ),
                        contentScale = ContentScale.Crop
                )
            }
        } else {
            Box(
                    modifier =
                            Modifier.size(100.dp)
                                    .clip(RoundedCornerShape(12.dp))
                                    .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center
            ) {
                Icon(
                        imageVector = Icons.Default.Image,
                        contentDescription = null,
                        modifier = Modifier.size(48.dp),
                        tint = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                )
            }
        }

        Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text(
                    text = personalObject.name,
                    style = MaterialTheme.typography.headlineSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurface
            )
            Text(
                    text = "ID: ${personalObject.id}",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.primary
            )
        }
    }
}

@Composable
private fun InfoSection(
        title: String,
        icon: androidx.compose.ui.graphics.vector.ImageVector,
        content: @Composable ColumnScope.() -> Unit
) {
    Card(
            modifier = Modifier.fillMaxWidth(),
            shape = RoundedCornerShape(12.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
            elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
                modifier = Modifier.padding(16.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                Icon(
                        imageVector = icon,
                        contentDescription = null,
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(20.dp)
                )
                Text(
                        text = title,
                        style = MaterialTheme.typography.titleMedium,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onSurface
                )
            }
            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            content()
        }
    }
}

@Composable
private fun InfoRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
                text = label,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Text(
                text = value,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
                maxLines = 1,
                overflow = TextOverflow.Ellipsis,
                modifier = Modifier.widthIn(max = 200.dp)
        )
    }
}

@Composable
private fun DebugImagesGallery(images: List<File>) {
    val scrollState = rememberScrollState()

    Row(
            modifier = Modifier.fillMaxWidth().horizontalScroll(scrollState),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        images.take(10).forEach { file -> DebugImageItem(file) }

        if (images.size > 10) {
            Box(
                    modifier =
                            Modifier.size(120.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(MaterialTheme.colorScheme.surfaceVariant),
                    contentAlignment = Alignment.Center
            ) {
                Text(
                        text = "+${images.size - 10}\nmore",
                        style = MaterialTheme.typography.bodyMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
private fun DebugImageItem(file: File) {
    var bitmap by
            remember(file.absolutePath) {
                mutableStateOf(BitmapFactory.decodeFile(file.absolutePath))
            }

    DisposableEffect(file.absolutePath) {
        onDispose {
            bitmap?.recycle()
            bitmap = null
        }
    }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        bitmap?.let {
            Image(
                    bitmap = it.asImageBitmap(),
                    contentDescription = file.name,
                    modifier =
                            Modifier.size(120.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .border(
                                            1.dp,
                                            MaterialTheme.colorScheme.outline,
                                            RoundedCornerShape(8.dp)
                                    ),
                    contentScale = ContentScale.Crop
            )
        }
                ?: Box(
                        modifier =
                                Modifier.size(120.dp)
                                        .clip(RoundedCornerShape(8.dp))
                                        .background(Color.Gray),
                        contentAlignment = Alignment.Center
                ) {
                    Icon(
                            imageVector = Icons.Default.BrokenImage,
                            contentDescription = null,
                            tint = Color.White
                    )
                }

        Text(
                text = file.name.take(15) + if (file.name.length > 15) "..." else "",
                style = MaterialTheme.typography.labelSmall,
                fontSize = 8.sp,
                color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.7f),
                maxLines = 1
        )
    }
}

// Data class for parsed embeddings info
private data class EmbeddingsInfo(
        val count: Int,
        val dimension: Int,
        val jsonSize: Int,
        val previewValues: String
)

private fun parseEmbeddingsInfo(embeddingJson: String): EmbeddingsInfo {
    return try {
        val jsonArray = JSONArray(embeddingJson)
        val count = jsonArray.length()

        var dimension = 0
        var previewValues = ""

        if (count > 0) {
            val firstEmbedding = jsonArray.getJSONArray(0)
            dimension = firstEmbedding.length()

            // Get first 5 values from first embedding
            val values = mutableListOf<String>()
            for (i in 0 until minOf(5, firstEmbedding.length())) {
                values.add(String.format("%.4f", firstEmbedding.getDouble(i)))
            }
            previewValues = "[${values.joinToString(", ")}...]"
        }

        EmbeddingsInfo(
                count = count,
                dimension = dimension,
                jsonSize = embeddingJson.length,
                previewValues = previewValues
        )
    } catch (e: Exception) {
        EmbeddingsInfo(
                count = 0,
                dimension = 0,
                jsonSize = embeddingJson.length,
                previewValues = "Error parsing: ${e.message}"
        )
    }
}

private fun findDebugImages(context: android.content.Context): List<File> {
    return try {
        val debugDir =
                File(
                        context.getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                        "Debug_Registration"
                )

        if (debugDir.exists() && debugDir.isDirectory) {
            debugDir.listFiles()
                    ?.filter { it.isFile && (it.extension == "jpg" || it.extension == "png") }
                    ?.sortedByDescending { it.lastModified() }
                    ?: emptyList()
        } else {
            emptyList()
        }
    } catch (e: Exception) {
        emptyList()
    }
}
