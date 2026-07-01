package com.florea_gabriel.impairedhelpapp.data.database

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface KnownPersonDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(person: KnownPerson): Long

    @Query("SELECT * FROM known_persons ORDER BY createdAt DESC")
    fun getAllPersons(): Flow<List<KnownPerson>>

    @Query("SELECT * FROM known_persons")
    suspend fun getAllPersonsList(): List<KnownPerson>

    @Delete
    suspend fun delete(person: KnownPerson)

    @Query("DELETE FROM known_persons WHERE id = :id")
    suspend fun deleteById(id: Long)

    @Query("SELECT COUNT(*) FROM known_persons")
    suspend fun getCount(): Int
}
