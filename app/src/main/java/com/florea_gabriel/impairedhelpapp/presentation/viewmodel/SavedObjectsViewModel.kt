package com.florea_gabriel.impairedhelpapp.presentation.viewmodel

import android.content.Context
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.florea_gabriel.impairedhelpapp.data.database.AppDatabase
import com.florea_gabriel.impairedhelpapp.data.database.PersonalObject
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

class SavedObjectsViewModel(context: Context) : ViewModel() {

    private val database = AppDatabase.getInstance(context)
    private val dao = database.personalObjectDao()

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading.asStateFlow()

    // Use stateIn to convert Flow to StateFlow
    val objects: StateFlow<List<PersonalObject>> = dao.getAllObjects()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    fun deleteObject(context: Context, personalObject: PersonalObject) {
        viewModelScope.launch {
            try {
                // Delete thumbnail file
                val thumbnailFile = File(personalObject.thumbnailPath)
                if (thumbnailFile.exists()) {
                    thumbnailFile.delete()
                }

                // Delete from database
                dao.delete(personalObject)
                // Flow will automatically update the objects list
            } catch (e: Exception) {
                android.util.Log.e("SavedObjectsViewModel", "Error deleting object", e)
            }
        }
    }

    // Removed searchObjects - using automatic Flow updates instead
}
