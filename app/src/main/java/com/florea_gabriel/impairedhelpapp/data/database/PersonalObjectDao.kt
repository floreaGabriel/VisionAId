package com.florea_gabriel.impairedhelpapp.data.database

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import kotlinx.coroutines.flow.Flow

/**
 * PersonalObjectDao: Data Access Object for PersonalObject entity.
 *
 * Provides methods to:
 * - Insert new personal objects
 * - Query all objects (for listing in UI)
 * - Search by name (fuzzy matching)
 * - Update last searched timestamp
 * - Delete objects
 */
@Dao
interface PersonalObjectDao {

    /**
     * Insert a new personal object.
     * @return The row ID of the inserted object
     */
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(personalObject: PersonalObject): Long

    /**
     * Get all personal objects, ordered by most recently searched.
     */
    @Query("SELECT * FROM personal_objects ORDER BY lastSearchedAt DESC, createdAt DESC")
    fun getAllObjects(): Flow<List<PersonalObject>>

    /**
     * Get all personal objects as a simple list (for searching).
     */
    @Query("SELECT * FROM personal_objects")
    suspend fun getAllObjectsList(): List<PersonalObject>

    /**
     * Find object by exact name (case-insensitive).
     */
    @Query("SELECT * FROM personal_objects WHERE LOWER(name) = LOWER(:name) LIMIT 1")
    suspend fun findByName(name: String): PersonalObject?

    /**
     * Find objects containing the search term in name (fuzzy search).
     */
    @Query("SELECT * FROM personal_objects WHERE LOWER(name) LIKE '%' || LOWER(:searchTerm) || '%'")
    suspend fun searchByName(searchTerm: String): List<PersonalObject>

    /**
     * Get object by ID.
     */
    @Query("SELECT * FROM personal_objects WHERE id = :id")
    suspend fun getById(id: Long): PersonalObject?

    /**
     * Update last searched timestamp.
     */
    @Query("UPDATE personal_objects SET lastSearchedAt = :timestamp WHERE id = :id")
    suspend fun updateLastSearched(id: Long, timestamp: Long = System.currentTimeMillis())

    /**
     * Update an existing object.
     */
    @Update
    suspend fun update(personalObject: PersonalObject)

    /**
     * Delete an object.
     */
    @Delete
    suspend fun delete(personalObject: PersonalObject)

    /**
     * Delete object by ID.
     */
    @Query("DELETE FROM personal_objects WHERE id = :id")
    suspend fun deleteById(id: Long)

    /**
     * Get count of saved objects.
     */
    @Query("SELECT COUNT(*) FROM personal_objects")
    suspend fun getCount(): Int
}
