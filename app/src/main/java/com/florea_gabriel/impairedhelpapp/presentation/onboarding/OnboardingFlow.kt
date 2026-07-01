package com.florea_gabriel.impairedhelpapp.presentation.onboarding

import android.content.Context
import android.speech.tts.TextToSpeech
import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInHorizontally
import androidx.compose.animation.slideOutHorizontally
import androidx.compose.animation.togetherWith
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material.icons.filled.AttachMoney
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Face
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material.icons.filled.Palette
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.florea_gabriel.impairedhelpapp.R
import java.util.Locale

// ============================================
// Persistence — first-launch flag
// ============================================
private const val PREFS_NAME = "visionaid_prefs"
private const val KEY_ONBOARDING_DONE = "onboarding_completed"

object OnboardingPrefs {
    fun isOnboardingCompleted(context: Context): Boolean =
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(KEY_ONBOARDING_DONE, false)

    fun markCompleted(context: Context) {
        context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(KEY_ONBOARDING_DONE, true)
            .apply()
    }
}

// ============================================
// Onboarding navigation
// ============================================
private enum class OnboardingStep { WELCOME, FEATURES }

// ============================================
// TTS narration — spoken aloud on each step (accessibility)
// ============================================
private const val WELCOME_NARRATION =
    "Bine ai venit în VisionAId, asistentul vizual inteligent pentru persoane cu " +
    "deficiențe de vedere. Aplicația detectează obstacole, recunoaște persoane, " +
    "îți găsește obiectele și identifică bancnote, totul pe dispozitiv, în limba română. " +
    "Apasă pe butonul Explore pentru a continua."

private const val FEATURES_NARRATION =
    "Când vei intra în aplicație, vei găsi cinci taburi în partea de jos a ecranului, " +
    "de la stânga la dreapta. " +
    "Primul tab, Cameră, detectează obstacolele din jur și îți descrie scena. " +
    "Al doilea tab, Obiecte, îți permite să înregistrezi și să găsești obiectele tale " +
    "personale cu ghidare prin realitate augmentată. " +
    "Al treilea tab, Culori, identifică culorile din jurul tău. " +
    "Al patrulea tab, Persoane, recunoaște fețele persoanelor cunoscute. " +
    "Al cincilea tab, Bani, detectează și numără bancnotele românești. " +
    "La final, apasă pe butonul din partea de jos a ecranului pentru a intra în " +
    "aplicație, pe primul tab, Cameră."

/**
 * OnboardingFlow: two-step intro shown only on first install.
 *
 * Step 1 (WELCOME): logo, app name, description, "Explore" button.
 * Step 2 (FEATURES): the 5 tabs presented as cards + "Intră în aplicație" button.
 *
 * When the user completes the flow, [onFinished] is called and the
 * completion flag is persisted so onboarding is skipped on future launches.
 */
@Composable
fun OnboardingFlow(onFinished: () -> Unit) {
    val context = LocalContext.current
    var step by remember { mutableStateOf(OnboardingStep.WELCOME) }

    // ── TTS — read each step aloud (Romanian) ─────────────────────────────────
    var tts by remember { mutableStateOf<TextToSpeech?>(null) }
    var ttsReady by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale("ro", "RO")
                tts?.setSpeechRate(1.3f)
                tts?.setPitch(0.8f)
                ttsReady = true
            }
        }
    }

    // Speak the narration for the current step once TTS is ready, and re-speak
    // whenever the step changes.
    LaunchedEffect(step, ttsReady) {
        if (!ttsReady) return@LaunchedEffect
        val narration = when (step) {
            OnboardingStep.WELCOME -> WELCOME_NARRATION
            OnboardingStep.FEATURES -> FEATURES_NARRATION
        }
        tts?.speak(narration, TextToSpeech.QUEUE_FLUSH, null, "onboarding_${step.name}")
    }

    // Stop and release TTS when leaving onboarding.
    DisposableEffect(Unit) {
        onDispose {
            tts?.stop()
            tts?.shutdown()
        }
    }

    Surface(modifier = Modifier.fillMaxSize(), color = Color(0xFF0F1419)) {
        AnimatedContent(
            targetState = step,
            transitionSpec = {
                (slideInHorizontally(tween(350)) { it } + fadeIn(tween(350))) togetherWith
                        (slideOutHorizontally(tween(350)) { -it } + fadeOut(tween(350)))
            },
            label = "onboardingTransition"
        ) { current ->
            when (current) {
                OnboardingStep.WELCOME -> WelcomeScreen(
                    onExplore = { step = OnboardingStep.FEATURES }
                )
                OnboardingStep.FEATURES -> FeaturesOverviewScreen(
                    onEnter = {
                        OnboardingPrefs.markCompleted(context)
                        onFinished()
                    }
                )
            }
        }
    }
}

// ============================================
// Screen 1 — Welcome
// ============================================
@Composable
private fun WelcomeScreen(onExplore: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF0F1419), Color(0xFF1A2332))
                )
            )
            .padding(horizontal = 32.dp, vertical = 48.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.SpaceBetween
    ) {
        // Top spacer for visual balance
        Spacer(Modifier.height(24.dp))

        // Logo + name + description block
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier.fillMaxWidth()
        ) {
            androidx.compose.foundation.Image(
                painter = painterResource(id = R.drawable.vision_ai_logo),
                contentDescription = "Logo VisionAId",
                modifier = Modifier
                    .size(200.dp)
                    .semantics { contentDescription = "Logo VisionAId" }
            )

            Spacer(Modifier.height(16.dp))

            Text(
                text = "VisionAId",
                fontSize = 44.sp,
                fontWeight = FontWeight.Bold,
                color = Color.White,
                textAlign = TextAlign.Center
            )

            Spacer(Modifier.height(20.dp))

            Text(
                text = "Asistent vizual inteligent pentru persoane cu deficiențe de vedere.",
                fontSize = 18.sp,
                color = Color.White.copy(alpha = 0.85f),
                textAlign = TextAlign.Center,
                lineHeight = 26.sp,
                modifier = Modifier.padding(horizontal = 8.dp)
            )

            Spacer(Modifier.height(16.dp))

            Text(
                text = "Detectează obstacole, recunoaște persoane, găsește-ți obiectele și identifică bancnote — totul pe dispozitiv, în limba română.",
                fontSize = 15.sp,
                color = Color.White.copy(alpha = 0.65f),
                textAlign = TextAlign.Center,
                lineHeight = 22.sp,
                modifier = Modifier.padding(horizontal = 8.dp)
            )
        }

        // Explore button
        Button(
            onClick = onExplore,
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .semantics { contentDescription = "Butonul Explore" },
            shape = RoundedCornerShape(18.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF0D9488),
                contentColor = Color.White
            )
        ) {
            Text(
                text = "Explore",
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold
            )
            Spacer(Modifier.size(10.dp))
            Icon(
                imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                contentDescription = null,
                modifier = Modifier.size(22.dp)
            )
        }
    }
}

// ============================================
// Screen 2 — Features overview
// ============================================
private data class FeatureCard(
    val title: String,
    val description: String,
    val icon: ImageVector,
    val tint: Color
)

private val featureList = listOf(
    FeatureCard(
        title = "Cameră",
        description = "Alerte de proximitate și descrierea scenei",
        icon = Icons.Default.CameraAlt,
        tint = Color(0xFF14B8A6)
    ),
    FeatureCard(
        title = "Obiecte",
        description = "Înregistrează și găsește obiectele tale cu ghidare AR",
        icon = Icons.Default.Inventory2,
        tint = Color(0xFF38BDF8)
    ),
    FeatureCard(
        title = "Culori",
        description = "Identifică culorile din jur",
        icon = Icons.Default.Palette,
        tint = Color(0xFFC4B5FD)
    ),
    FeatureCard(
        title = "Persoane",
        description = "Recunoaștere facială din opt unghiuri",
        icon = Icons.Default.Face,
        tint = Color(0xFFFBBF24)
    ),
    FeatureCard(
        title = "Bani",
        description = "Detecție și numărare bancnote românești",
        icon = Icons.Default.AttachMoney,
        tint = Color(0xFF22C55E)
    )
)

@Composable
private fun FeaturesOverviewScreen(onEnter: () -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(Color(0xFF0F1419), Color(0xFF1A2332))
                )
            )
            .padding(horizontal = 24.dp, vertical = 32.dp)
    ) {
        Text(
            text = "Funcționalități",
            fontSize = 32.sp,
            fontWeight = FontWeight.Bold,
            color = Color.White,
            modifier = Modifier.padding(bottom = 8.dp)
        )

        Text(
            text = "Cele 5 module ale aplicației",
            fontSize = 15.sp,
            color = Color.White.copy(alpha = 0.65f),
            modifier = Modifier.padding(bottom = 20.dp)
        )

        // Scrollable feature cards
        Column(
            modifier = Modifier
                .weight(1f)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            featureList.forEach { feature ->
                FeatureCardItem(feature)
            }
        }

        Spacer(Modifier.height(20.dp))

        // Enter app button
        Button(
            onClick = onEnter,
            modifier = Modifier
                .fillMaxWidth()
                .height(64.dp)
                .semantics { contentDescription = "Butonul Intră în aplicație" },
            shape = RoundedCornerShape(18.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = Color(0xFF0D9488),
                contentColor = Color.White
            )
        ) {
            Text(
                text = "Intră în aplicație",
                fontSize = 20.sp,
                fontWeight = FontWeight.SemiBold
            )
        }
    }
}

@Composable
private fun FeatureCardItem(feature: FeatureCard) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .semantics {
                contentDescription = "${feature.title}: ${feature.description}"
            },
        shape = RoundedCornerShape(16.dp),
        color = Color(0xFF1E293B)
    ) {
        Row(
            modifier = Modifier.padding(16.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            // Icon circle
            Box(
                modifier = Modifier
                    .size(52.dp)
                    .clip(CircleShape)
                    .background(feature.tint.copy(alpha = 0.18f)),
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = feature.icon,
                    contentDescription = null,
                    tint = feature.tint,
                    modifier = Modifier.size(28.dp)
                )
            }

            Spacer(Modifier.size(16.dp))

            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = feature.title,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = Color.White
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    text = feature.description,
                    fontSize = 14.sp,
                    color = Color.White.copy(alpha = 0.7f),
                    lineHeight = 18.sp
                )
            }
        }
    }
}
