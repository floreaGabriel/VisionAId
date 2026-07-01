package com.florea_gabriel.impairedhelpapp.data.database

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase

/**
 * AppDatabase: Room database for the VisionAId.
 *
 * Contains:
 * - PersonalObject table: Stores user's personal objects with embeddings for searching
 *
 * The database is created as a singleton to ensure only one instance exists.
 */
@Database(entities = [PersonalObject::class, KnownPerson::class], version = 5, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {

    abstract fun personalObjectDao(): PersonalObjectDao
    abstract fun knownPersonDao(): KnownPersonDao

    companion object {
        private const val DATABASE_NAME = "impaired_help_db"

        @Volatile private var INSTANCE: AppDatabase? = null

        // Migration 3 -> 4: Add embeddingBlob column for binary storage
        private val MIGRATION_3_4 =
                object : Migration(3, 4) {
                    override fun migrate(db: SupportSQLiteDatabase) {
                        db.execSQL(
                                "ALTER TABLE personal_objects ADD COLUMN embeddingBlob BLOB DEFAULT NULL"
                        )
                    }
                }

        // Migration 4 -> 5: Add known_persons table for face recognition
        private val MIGRATION_4_5 =
                object : Migration(4, 5) {
                    override fun migrate(db: SupportSQLiteDatabase) {
                        db.execSQL(
                                """CREATE TABLE IF NOT EXISTS known_persons (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
                                    name TEXT NOT NULL,
                                    embeddingBlob BLOB NOT NULL,
                                    thumbnailPath TEXT NOT NULL,
                                    createdAt INTEGER NOT NULL DEFAULT 0
                                )"""
                        )
                    }
                }

        /** Get the singleton database instance. Uses double-checked locking for thread safety. */
        fun getInstance(context: Context): AppDatabase {
            return INSTANCE
                    ?: synchronized(this) {
                        INSTANCE ?: buildDatabase(context).also { INSTANCE = it }
                    }
        }

        private fun buildDatabase(context: Context): AppDatabase {
            return Room.databaseBuilder(
                            context.applicationContext,
                            AppDatabase::class.java,
                            DATABASE_NAME
                    )
                    .addMigrations(MIGRATION_3_4, MIGRATION_4_5)
                    .fallbackToDestructiveMigration()
                    .build()
        }
    }
}
