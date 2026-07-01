package com.florea_gabriel.impairedhelpapp.data.database

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

/**
 * PersonalObject: Entity representing a user's personal object stored for future searching.
 *
 * When a user saves a personal object:
 * 1. They take a photo of the object
 * 2. The app extracts visual features (embedding) from the photo
 * 3. The embedding is stored as a JSON string in this entity
 * 4. When searching, live camera frames are compared against this embedding
 *
 * @property id Auto-generated unique identifier
 * @property name User-given name for the object (e.g., "portofelul meu", "cheile de casă")
 * @property description Optional description for additional context
 * @property embeddingJson Visual feature embedding stored as JSON array string (kept for debug)
 * @property embeddingBlob Binary blob storage for embeddings (2.5x smaller, 150x faster parse)
 * @property thumbnailPath Path to saved thumbnail image for UI display
 * @property createdAt Timestamp when object was saved
 * @property lastSearchedAt Timestamp of last search (for sorting by recent)
 */
@Entity(tableName = "personal_objects")
data class PersonalObject(
        @PrimaryKey(autoGenerate = true) val id: Long = 0,
        val name: String,
        val description: String = "",

        // Embedding stored as JSON string: "[[0.123, 0.456, ...], [...]]"
        // Using String instead of FloatArray for Room compatibility
        // KEPT for backward compatibility and debugging
        val embeddingJson: String,

        // Binary blob storage for embeddings (PREFERRED - faster parsing)
        // Format: [count:Int32][embedding1:512xFloat32][embedding2:512xFloat32]...
        @ColumnInfo(typeAffinity = ColumnInfo.BLOB) val embeddingBlob: ByteArray? = null,

        // Path to thumbnail image in app's internal storage
        val thumbnailPath: String,
        val createdAt: Long = System.currentTimeMillis(),
        val lastSearchedAt: Long = 0,

        // Gemini 2.0 Metadata - For hybrid search
        val geminiLabel: String? = null, // e.g. "Wallet"
        val geminiDescription: String? = null, // e.g. "Black leather wallet with zipper"
        val detectionKeywords: String? = null, // e.g. "wallet,bag,handbag" (for YOLO search)

        // Adaptive threshold calculated during registration (fallback: 0.65)
        val recommendedThreshold: Float? = null
) {
        // Override equals/hashCode to handle ByteArray comparison
        override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (javaClass != other?.javaClass) return false
                other as PersonalObject
                if (id != other.id) return false
                if (name != other.name) return false
                if (embeddingBlob != null) {
                        if (other.embeddingBlob == null) return false
                        if (!embeddingBlob.contentEquals(other.embeddingBlob)) return false
                } else if (other.embeddingBlob != null) return false
                return true
        }

        override fun hashCode(): Int {
                var result = id.hashCode()
                result = 31 * result + name.hashCode()
                result = 31 * result + (embeddingBlob?.contentHashCode() ?: 0)
                return result
        }
}
