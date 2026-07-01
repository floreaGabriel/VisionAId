package com.florea_gabriel.impairedhelpapp.presentation.viewmodel

import android.content.Context
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.florea_gabriel.impairedhelpapp.data.database.AppDatabase
import com.florea_gabriel.impairedhelpapp.data.database.KnownPerson
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

class PersonsViewModel(context: Context) : ViewModel() {

    private val database = AppDatabase.getInstance(context)
    private val dao = database.knownPersonDao()

    val persons: StateFlow<List<KnownPerson>> = dao.getAllPersons()
        .stateIn(
            scope = viewModelScope,
            started = SharingStarted.WhileSubscribed(5000),
            initialValue = emptyList()
        )

    fun deletePerson(context: Context, person: KnownPerson) {
        viewModelScope.launch {
            try {
                val thumbnailFile = File(person.thumbnailPath)
                if (thumbnailFile.exists()) {
                    thumbnailFile.delete()
                }
                dao.delete(person)
            } catch (e: Exception) {
                Log.e("PersonsViewModel", "Error deleting person", e)
            }
        }
    }
}
