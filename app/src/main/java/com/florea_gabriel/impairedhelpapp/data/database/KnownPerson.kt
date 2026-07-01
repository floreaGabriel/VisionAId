package com.florea_gabriel.impairedhelpapp.data.database

import androidx.room.ColumnInfo
import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "known_persons")
data class KnownPerson(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val name: String,
    @ColumnInfo(typeAffinity = ColumnInfo.BLOB) val embeddingBlob: ByteArray,
    val thumbnailPath: String,
    val createdAt: Long = System.currentTimeMillis()
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as KnownPerson
        if (id != other.id) return false
        if (name != other.name) return false
        if (!embeddingBlob.contentEquals(other.embeddingBlob)) return false
        return true
    }

    override fun hashCode(): Int {
        var result = id.hashCode()
        result = 31 * result + name.hashCode()
        result = 31 * result + embeddingBlob.contentHashCode()
        return result
    }
}
