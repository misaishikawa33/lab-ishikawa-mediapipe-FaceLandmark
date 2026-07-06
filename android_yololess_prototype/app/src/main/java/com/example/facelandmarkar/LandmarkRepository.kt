package com.example.facelandmarkar

import android.content.Context
import org.opencv.core.Point3

data class Landmark3D(
    val id: Int,
    val point: Point3
)

class LandmarkRepository(private val context: Context) {
    fun load3dLandmarks(assetName: String = "face_3d_points.csv"): Map<Int, Point3> {
        val result = linkedMapOf<Int, Point3>()
        context.assets.open(assetName).bufferedReader().useLines { lines ->
            lines.drop(1).forEach { line ->
                val cols = line.split(",")
                if (cols.size < 4) {
                    return@forEach
                }
                val id = cols[0].trim().toInt()
                val x = cols[1].trim().toDouble()
                val y = cols[2].trim().toDouble()
                val z = cols[3].trim().toDouble()
                result[id] = Point3(x, y, z)
            }
        }
        return result
    }
}
