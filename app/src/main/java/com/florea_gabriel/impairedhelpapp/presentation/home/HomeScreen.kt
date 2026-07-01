package com.florea_gabriel.impairedhelpapp.presentation.home

import android.Manifest
import android.speech.tts.TextToSpeech
import android.util.Log
import androidx.activity.compose.BackHandler
import kotlinx.coroutines.delay
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.ArrowBack
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.AttachMoney
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material3.FloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationBarItemDefaults
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.florea_gabriel.impairedhelpapp.presentation.add_object.AdvancedAddObjectScreen
import com.florea_gabriel.impairedhelpapp.presentation.color.ColorDetectionScreen
import com.florea_gabriel.impairedhelpapp.presentation.money.MoneyDetectionScreen
import com.florea_gabriel.impairedhelpapp.presentation.persons.FaceRegistrationScreen
import com.florea_gabriel.impairedhelpapp.presentation.persons.LiveFaceRecognitionScreen
import com.florea_gabriel.impairedhelpapp.presentation.persons.PersonsListScreen
import com.florea_gabriel.impairedhelpapp.presentation.saved_objects.ObjectDetailsScreen
import com.florea_gabriel.impairedhelpapp.presentation.saved_objects.SavedObjectsScreen
import com.florea_gabriel.impairedhelpapp.presentation.search.ARSearchScreen
import com.florea_gabriel.impairedhelpapp.presentation.viewmodel.PersonsViewModel
import com.florea_gabriel.impairedhelpapp.presentation.viewmodel.SavedObjectsViewModel
import com.florea_gabriel.impairedhelpapp.utils.VoiceCommandManager
import java.util.Locale

/**
 * HomeScreen: Main navigation host with a 2-tab Bottom Navigation.
 *
 * Tab 0 — Camera: LiveDetectionScreen, always kept in composition so the camera
 *   never restarts. ML processing is paused when this tab is hidden.
 *
 * Tab 1 — Obiecte: SavedObjectsScreen with sub-navigation (LIST → ADD / DETAILS / SEARCH).
 *   When ADD or SEARCH is active, the Camera layer is removed from composition to avoid
 *   conflicts with their own camera/AR sessions.
 */
@Composable
fun HomeScreen() {
    val context = LocalContext.current
    val view = LocalView.current

    // Keep the screen on while the app is active — camera + AI apps need this
    DisposableEffect(Unit) {
        view.keepScreenOn = true
        onDispose { view.keepScreenOn = false }
    }

    // ── Tab navigation ──────────────────────────────────────────────────────────
    var selectedTab by rememberSaveable { mutableIntStateOf(0) }

    // ── Objects sub-navigation ──────────────────────────────────────────────────
    var objectsSubScreen by rememberSaveable { mutableStateOf(ObjectsSubScreen.LIST) }
    var selectedObjectId by remember { mutableStateOf<Long?>(null) }
    var searchTarget by remember { mutableStateOf<String?>(null) }

    // ── Persons sub-navigation ───────────────────────────────────────────────────
    var personsSubScreen by rememberSaveable { mutableStateOf(PersonsSubScreen.LIVE) }

    // ── Scene description ───────────────────────────────────────────────────────
    var triggerSceneDescription by remember { mutableStateOf(false) }

    // ── TTS ────────────────────────────────────────────────────────────────────
    var textToSpeech by remember { mutableStateOf<TextToSpeech?>(null) }

    // ── ViewModels ─────────────────────────────────────────────────────────────
    val savedObjectsViewModel = remember { SavedObjectsViewModel(context) }
    val allSavedObjects by savedObjectsViewModel.objects.collectAsState()
    val personsViewModel = remember { PersonsViewModel(context) }

    // ── Voice ──────────────────────────────────────────────────────────────────
    var isListening by remember { mutableStateOf(false) }
    var hasAudioPermission by remember { mutableStateOf(false) }
    var ttsReady by remember { mutableStateOf(false) }

    // Initialize TTS
    LaunchedEffect(Unit) {
        textToSpeech = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale("ro", "RO")
                textToSpeech?.setSpeechRate(1.15f)
                textToSpeech?.setPitch(0.8f)
                ttsReady = true
            }
        }
    }

    // Helper: speak only if TTS is ready
    val safeTtsSpeak: (String) -> Unit = { text ->
        if (ttsReady) {
            textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, null)
        }
    }

    // Voice Command Manager
    val voiceCommandManager = remember {
        VoiceCommandManager(context) { command ->
            Log.d("HomeScreen", "Voice command: $command")
            when (command) {
                VoiceCommandManager.VoiceCommand.DESCRIBE_SCENE -> {
                    triggerSceneDescription = true
                    selectedTab = 0
                    safeTtsSpeak("Analizez scena")
                }
                VoiceCommandManager.VoiceCommand.STOP_SEARCH -> {
                    searchTarget = null
                    if (objectsSubScreen == ObjectsSubScreen.SEARCH) {
                        objectsSubScreen = ObjectsSubScreen.LIST
                    }
                    safeTtsSpeak("Căutare oprită")
                }
                VoiceCommandManager.VoiceCommand.GO_CAMERA -> {
                    selectedTab = 0
                    safeTtsSpeak("Cameră")
                }
                VoiceCommandManager.VoiceCommand.GO_OBJECTS -> {
                    selectedTab = 1
                    objectsSubScreen = ObjectsSubScreen.LIST
                    safeTtsSpeak("Obiectele mele")
                }
                VoiceCommandManager.VoiceCommand.GO_COLORS -> {
                    selectedTab = 2
                    safeTtsSpeak("Detecție culori")
                }
                VoiceCommandManager.VoiceCommand.GO_PERSONS -> {
                    selectedTab = 3
                    personsSubScreen = PersonsSubScreen.LIVE
                    safeTtsSpeak("Recunoaștere persoane")
                }
                VoiceCommandManager.VoiceCommand.GO_MONEY -> {
                    selectedTab = 4
                    safeTtsSpeak("Detecție bani")
                }
                else -> { /* handled by specific callbacks below */ }
            }
        }
    }

    // Setup voice callbacks
    LaunchedEffect(voiceCommandManager) {
        voiceCommandManager.setOnListeningStarted { isListening = true }
        voiceCommandManager.setOnListeningEnded { isListening = false }
        voiceCommandManager.setOnError { isListening = false }

        // "find X" → switch to objects tab → AR search
        voiceCommandManager.setOnFindObjectCommand { objectName ->
            searchTarget = objectName
            selectedTab = 1
            objectsSubScreen = ObjectsSubScreen.SEARCH
        }

        // "save X" → go to objects tab → add screen
        voiceCommandManager.setOnSaveObjectCommand { _ ->
            selectedTab = 1
            objectsSubScreen = ObjectsSubScreen.ADD
        }
    }

    // Audio permission launcher
    val audioPermissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasAudioPermission = isGranted
        if (isGranted) {
            voiceCommandManager.startListening()
        } else {
            safeTtsSpeak("Permisiune microfon necesară")
        }
    }

    val onMicPress = {
        if (!ttsReady) {
            // TTS not ready yet, ignore mic press
            Log.d("HomeScreen", "Mic press ignored — TTS not ready")
        } else if (!hasAudioPermission) {
            audioPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        } else {
            voiceCommandManager.startListening()
        }
    }

    val onMicRelease = {
        if (isListening) {
            voiceCommandManager.stopListening()
        }
    }

    // Cleanup
    DisposableEffect(Unit) {
        onDispose {
            voiceCommandManager.release()
            textToSpeech?.stop()
            textToSpeech?.shutdown()
        }
    }

    // Leaving AR search is a TWO-PHASE process, because abruptly removing ARSearchScreen
    // crashes natively in two ways: (1) Filament aborts when its engine is destroyed with
    // the AR marker still attached; (2) the main camera's ONNX session is hit while being
    // torn down. So: `searchExiting` asks ARSearchScreen to dismantle its AR scene first
    // (it keeps rendering meanwhile); when done it calls back and we actually navigate.
    var searchExiting by remember { mutableStateOf(false) }
    // After navigation, keep the main camera out of composition for a short settling
    // window so its CameraX + ONNX restart doesn't overlap the ARCore release.
    var searchTeardownActive by remember { mutableStateOf(false) }
    LaunchedEffect(searchTeardownActive) {
        if (searchTeardownActive) {
            delay(700)
            searchTeardownActive = false
        }
    }

    // Phase 1: request graceful AR teardown (does NOT navigate yet). Idempotent.
    val requestExitSearch: () -> Unit = {
        if (objectsSubScreen == ObjectsSubScreen.SEARCH) searchExiting = true
    }
    // Phase 2: ARSearchScreen finished dismantling — now safe to leave the screen.
    val onSearchTornDown: () -> Unit = {
        searchExiting = false
        searchTarget = null
        searchTeardownActive = true
        objectsSubScreen = ObjectsSubScreen.LIST
    }

    // ── BackHandler for objects sub-navigation ──────────────────────────────────
    BackHandler(enabled = selectedTab == 1 && objectsSubScreen != ObjectsSubScreen.LIST) {
        when (objectsSubScreen) {
            ObjectsSubScreen.ADD -> objectsSubScreen = ObjectsSubScreen.LIST
            ObjectsSubScreen.DETAILS -> {
                selectedObjectId = null
                objectsSubScreen = ObjectsSubScreen.LIST
            }
            ObjectsSubScreen.SEARCH -> requestExitSearch()
            else -> { /* LIST: system handles back (exits app) */ }
        }
    }

    // ── BackHandler for persons sub-navigation ───────────────────────────────────
    BackHandler(enabled = selectedTab == 3 && personsSubScreen != PersonsSubScreen.LIVE) {
        when (personsSubScreen) {
            PersonsSubScreen.LIST, PersonsSubScreen.REGISTER -> personsSubScreen = PersonsSubScreen.LIVE
            else -> {}
        }
    }

    // ── Visibility helpers ──────────────────────────────────────────────────────
    // Camera must leave composition during ADD/SEARCH/DEBUG or persons ADD (they have their own camera session)
    val cameraInComposition = selectedTab != 2 &&
            selectedTab != 3 &&
            selectedTab != 4 &&
            objectsSubScreen != ObjectsSubScreen.ADD &&
            objectsSubScreen != ObjectsSubScreen.SEARCH &&
            !searchTeardownActive

    // Top bar title per module
    val topBarTitle = when (selectedTab) {
        0 -> "Cameră"
        1 -> when (objectsSubScreen) {
            ObjectsSubScreen.LIST -> "Obiectele mele"
            ObjectsSubScreen.DETAILS -> "Detalii obiect"
            ObjectsSubScreen.ADD -> "Înregistrare obiect"
            ObjectsSubScreen.SEARCH -> "Căutare obiect"
        }
        2 -> "Detecție culori"
        3 -> when (personsSubScreen) {
            PersonsSubScreen.LIVE -> "Recunoaștere facială"
            PersonsSubScreen.LIST -> "Persoanele mele"
            PersonsSubScreen.REGISTER -> "Înregistrare persoană"
        }
        4 -> "Detecție bani"
        else -> ""
    }

    // Show back button on sub-screens
    val showBackButton = when (selectedTab) {
        1 -> objectsSubScreen != ObjectsSubScreen.LIST
        3 -> personsSubScreen != PersonsSubScreen.LIVE
        else -> false
    }

    val onTopBarBack: () -> Unit = {
        when (selectedTab) {
            1 -> when (objectsSubScreen) {
                ObjectsSubScreen.ADD -> objectsSubScreen = ObjectsSubScreen.LIST
                ObjectsSubScreen.DETAILS -> {
                    selectedObjectId = null
                    objectsSubScreen = ObjectsSubScreen.LIST
                }
                ObjectsSubScreen.SEARCH -> requestExitSearch()
                else -> {}
            }
            3 -> personsSubScreen = PersonsSubScreen.LIVE
        }
    }

    // Smooth alpha for camera layer
    val cameraAlpha by animateFloatAsState(
        targetValue = if (selectedTab == 0) 1f else 0f,
        animationSpec = tween(durationMillis = 250),
        label = "cameraAlpha"
    )

    // ── Root layout ─────────────────────────────────────────────────────────────
    Box(modifier = Modifier.fillMaxSize()) {

        // ── Camera layer (always alive unless ADD or SEARCH is open) ────────────
        if (cameraInComposition) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .alpha(cameraAlpha)
            ) {
                LiveDetectionScreen(
                    isScreenVisible = selectedTab == 0,
                    triggerSceneDescription = triggerSceneDescription && selectedTab == 0,
                    onSceneDescriptionComplete = { triggerSceneDescription = false },
                    onSceneDescription = { description ->
                        textToSpeech?.speak(description, TextToSpeech.QUEUE_FLUSH, null, "scene_desc")
                    }
                )
            }
        }

        // ── Objects tab content (only composed when tab 1 is selected) ──────────
        if (selectedTab == 1) {
            when (objectsSubScreen) {
                ObjectsSubScreen.LIST -> {
                    SavedObjectsScreen(
                        viewModel = savedObjectsViewModel,
                        onBackClick = { /* no back from tab root */ },
                        showBackButton = false,
                        bottomPadding = 210.dp,
                        onAddObjectClick = { objectsSubScreen = ObjectsSubScreen.ADD },
                        onSearchClick = { objectName ->
                            searchTarget = objectName
                            objectsSubScreen = ObjectsSubScreen.SEARCH
                        },
                        onDetailsClick = { objectId ->
                            selectedObjectId = objectId
                            objectsSubScreen = ObjectsSubScreen.DETAILS
                        }
                    )
                }

                ObjectsSubScreen.ADD -> {
                    // Camera is NOT in composition here (cameraInComposition = false)
                    AdvancedAddObjectScreen(
                        onObjectSaved = {
                            objectsSubScreen = ObjectsSubScreen.LIST
                        },
                        onBackClick = { objectsSubScreen = ObjectsSubScreen.LIST }
                    )
                }

                ObjectsSubScreen.DETAILS -> {
                    val personalObject = allSavedObjects.find { it.id == selectedObjectId }
                    if (personalObject != null) {
                        ObjectDetailsScreen(
                            personalObject = personalObject,
                            onBackClick = {
                                selectedObjectId = null
                                objectsSubScreen = ObjectsSubScreen.LIST
                            }
                        )
                    } else {
                        // Object not found (e.g. deleted) — go back to list
                        LaunchedEffect(Unit) {
                            selectedObjectId = null
                            objectsSubScreen = ObjectsSubScreen.LIST
                        }
                    }
                }

                ObjectsSubScreen.SEARCH -> {
                    // Camera is NOT in composition here (cameraInComposition = false)
                    val objectToSearch = searchTarget ?: ""
                    if (objectToSearch.isNotEmpty()) {
                        ARSearchScreen(
                            objectName = objectToSearch,
                            // Two-phase exit: requestExitSearch triggers graceful AR teardown
                            // inside ARSearchScreen; onSearchTornDown then navigates away.
                            shuttingDown = searchExiting,
                            onArrived = requestExitSearch,
                            onTornDown = onSearchTornDown
                        )
                    } else {
                        LaunchedEffect(Unit) { objectsSubScreen = ObjectsSubScreen.LIST }
                    }
                }
            }
        }

        // ── Color detection tab (only composed when tab 2 is selected) ────────
        if (selectedTab == 2) {
            ColorDetectionScreen(
                isScreenVisible = true,
                bottomPadding = 210.dp,
                onColorDetected = { detectedColor ->
                    textToSpeech?.speak(detectedColor, TextToSpeech.QUEUE_FLUSH, null, "color")
                }
            )
        }

        // ── Persons tab content (only composed when tab 3 is selected) ───────
        if (selectedTab == 3) {
            when (personsSubScreen) {
                PersonsSubScreen.LIVE -> {
                    LiveFaceRecognitionScreen(
                        isScreenVisible = selectedTab == 3,
                        personsViewModel = personsViewModel,
                        bottomPadding = 210.dp,
                        onSpeakRequest = { text ->
                            textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "face_tts")
                        },
                        onShowPersonsList = { personsSubScreen = PersonsSubScreen.LIST },
                        onStartRegistration = { personsSubScreen = PersonsSubScreen.REGISTER }
                    )
                }
                PersonsSubScreen.LIST -> {
                    PersonsListScreen(
                        viewModel = personsViewModel,
                        onAddPersonClick = { personsSubScreen = PersonsSubScreen.REGISTER },
                        onBackClick = { personsSubScreen = PersonsSubScreen.LIVE },
                        bottomPadding = 210.dp
                    )
                }
                PersonsSubScreen.REGISTER -> {
                    FaceRegistrationScreen(
                        onRegistrationComplete = { _ ->
                            // TTS already spoken by FaceRegistrationScreen before callback
                            personsSubScreen = PersonsSubScreen.LIVE
                        },
                        onBackClick = { personsSubScreen = PersonsSubScreen.LIVE },
                        onSpeakRequest = { text ->
                            textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "reg_tts")
                        }
                    )
                }
            }
        }

        // ── Money detection tab (only composed when tab 4 is selected) ────────
        if (selectedTab == 4) {
            MoneyDetectionScreen(
                isScreenVisible = true,
                bottomPadding = 210.dp,
                onSpeakRequest = { text ->
                    textToSpeech?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "money_tts")
                }
            )
        }

        // ── Top Bar ───────────────────────────────────────────────────────────────
        AppTopBar(
            title = topBarTitle,
            showBack = showBackButton,
            onBackClick = onTopBarBack,
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
        )

        // ── Hold-to-talk FAB (hidden during ADD/SEARCH/REGISTER) ─────────────
        val showMic = when (selectedTab) {
            1 -> objectsSubScreen == ObjectsSubScreen.LIST || objectsSubScreen == ObjectsSubScreen.DETAILS
            3 -> personsSubScreen == PersonsSubScreen.LIVE || personsSubScreen == PersonsSubScreen.LIST
            else -> true
        }
        if (showMic) {
            Surface(
                color = if (isListening) Color(0xFFE53935)
                        else Color.DarkGray.copy(alpha = 0.85f),
                shape = CircleShape,
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .padding(bottom = 90.dp)
                    .size(112.dp)
                    .pointerInput(hasAudioPermission) {
                        detectTapGestures(
                            onPress = {
                                onMicPress()
                                tryAwaitRelease()
                                onMicRelease()
                            }
                        )
                    }
                    .semantics {
                        contentDescription =
                            if (isListening) "Ascult... eliberează pentru a opri"
                            else "Ține apăsat pentru comandă vocală"
                    }
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    Icon(
                        imageVector = Icons.Default.Mic,
                        contentDescription = null,
                        tint = Color.White,
                        modifier = Modifier.size(48.dp)
                    )
                }
            }

            // ── Add object "+" FAB (only on objects LIST) ────────────────────────
            if (selectedTab == 1 && objectsSubScreen == ObjectsSubScreen.LIST) {
                Surface(
                    color = MaterialTheme.colorScheme.primary,
                    shape = CircleShape,
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .padding(bottom = 110.dp, start = 265.dp)
                        .size(56.dp)
                        .clickable { objectsSubScreen = ObjectsSubScreen.ADD }
                        .semantics {
                            contentDescription = "Adaugă un obiect nou"
                        }
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Icon(
                            imageVector = Icons.Default.Add,
                            contentDescription = null,
                            tint = Color.White,
                            modifier = Modifier.size(28.dp)
                        )
                    }
                }
            }
        }

        // ── Bottom Navigation Bar ──────────────────────────────────────────────
        AppBottomNavigationBar(
                selectedTab = selectedTab,
                onTabSelected = { tab ->
                    selectedTab = tab
                    // Reset sub-screens when switching tabs
                    if (tab == 1 && objectsSubScreen != ObjectsSubScreen.LIST) {
                        objectsSubScreen = ObjectsSubScreen.LIST
                    }
                    if (tab == 3 && personsSubScreen != PersonsSubScreen.LIVE) {
                        personsSubScreen = PersonsSubScreen.LIVE
                    }
                },
                modifier = Modifier
                    .align(Alignment.BottomCenter)
                    .fillMaxWidth()
            )
    }
}

// ── Objects tab sub-screens ─────────────────────────────────────────────────────

private enum class ObjectsSubScreen { LIST, ADD, DETAILS, SEARCH }
private enum class PersonsSubScreen { LIVE, LIST, REGISTER }

// ── Bottom Navigation Bar ──────────────────────────────────────────────────────

private data class NavTab(
    val label: String,
    val icon: ImageVector,
    val contentDescription: String
)

private val NAV_TABS = listOf(
    NavTab("Cameră", Icons.Default.CameraAlt, "Tabul camerei"),
    NavTab("Obiecte", Icons.Default.Inventory2, "Tabul obiectelor salvate"),
    NavTab("Culori", Icons.Default.Palette, "Tabul detecției de culori"),
    NavTab("Persoane", Icons.Default.Face, "Tabul persoanelor"),
    NavTab("Bani", Icons.Default.AttachMoney, "Tabul detectiei de bani")
)

@Composable
private fun AppTopBar(
    title: String,
    showBack: Boolean = false,
    onBackClick: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    Surface(
        modifier = modifier.height(48.dp),
        color = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp
    ) {
        Box(
            modifier = Modifier.fillMaxSize(),
            contentAlignment = Alignment.CenterStart
        ) {
            if (showBack) {
                IconButton(onClick = onBackClick) {
                    Icon(
                        imageVector = Icons.Default.ArrowBack,
                        contentDescription = "Înapoi",
                        tint = MaterialTheme.colorScheme.onSurface
                    )
                }
            }
            Text(
                text = title,
                fontSize = 18.sp,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
                modifier = Modifier.padding(start = if (showBack) 48.dp else 16.dp)
            )
        }
    }
}

@Composable
private fun AppBottomNavigationBar(
    selectedTab: Int,
    onTabSelected: (Int) -> Unit,
    modifier: Modifier = Modifier
) {
    NavigationBar(
        modifier = modifier,
        containerColor = MaterialTheme.colorScheme.surface,
        tonalElevation = 3.dp
    ) {
        NAV_TABS.forEachIndexed { index, tab ->
            NavigationBarItem(
                selected = selectedTab == index,
                onClick = { onTabSelected(index) },
                icon = {
                    Icon(
                        imageVector = tab.icon,
                        contentDescription = tab.contentDescription
                    )
                },
                label = { Text(tab.label) },
                colors = NavigationBarItemDefaults.colors(
                    selectedIconColor = MaterialTheme.colorScheme.primary,
                    selectedTextColor = MaterialTheme.colorScheme.primary,
                    indicatorColor = MaterialTheme.colorScheme.primaryContainer,
                    unselectedIconColor = MaterialTheme.colorScheme.onSurfaceVariant,
                    unselectedTextColor = MaterialTheme.colorScheme.onSurfaceVariant
                )
            )
        }
    }
}
