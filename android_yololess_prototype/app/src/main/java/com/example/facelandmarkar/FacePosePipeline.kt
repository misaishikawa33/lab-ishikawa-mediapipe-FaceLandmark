package com.example.facelandmarkar

import org.opencv.core.Point
import org.opencv.core.Point3

data class Landmark2D(
    val id: Int,
    val x: Double,
    val y: Double
)

class FacePosePipeline(
    private val landmarks3d: Map<Int, Point3>,
    private val poseEstimator: PoseEstimator
) {
    private val eyePointIds = listOf(
        6, 7, 8, 9, 10, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        33, 34, 35, 46, 52, 53, 54, 55, 56, 63, 65, 66, 67, 68,
        69, 70, 71, 103, 104, 105, 107, 108, 109, 110, 112, 113,
        122, 124, 127, 130, 133, 139, 143, 144, 145, 151, 153, 154,
        155, 156, 157, 158, 159, 160, 161, 162, 163, 168, 173, 189,
        190, 193, 221, 222, 223, 224, 225, 226, 243, 244, 245, 246,
        247, 249, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260,
        263, 264, 265, 276, 282, 283, 284, 285, 286, 293, 295, 296,
        297, 298, 299, 300, 301, 332, 333, 334, 336, 337, 338, 339,
        341, 342, 351, 353, 356, 359, 362, 368, 372, 373, 374, 380,
        381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 398, 413,
        414, 417, 441, 442, 443, 444, 445, 446, 463, 464, 465, 466,
        467
    )

    fun estimateFromEyePoints(
        landmarks2d: List<Landmark2D>,
        imageWidth: Int,
        imageHeight: Int
    ): PoseResult? {
        return estimate(landmarks2d, eyePointIds, imageWidth, imageHeight)
    }

    fun estimateFromAllPoints(
        landmarks2d: List<Landmark2D>,
        imageWidth: Int,
        imageHeight: Int
    ): PoseResult? {
        val ids = landmarks3d.keys.sorted()
        return estimate(landmarks2d, ids, imageWidth, imageHeight)
    }

    private fun estimate(
        landmarks2d: List<Landmark2D>,
        targetIds: List<Int>,
        imageWidth: Int,
        imageHeight: Int
    ): PoseResult? {
        val landmarks2dById = landmarks2d.associateBy { it.id }
        val objectPoints = mutableListOf<Point3>()
        val imagePoints = mutableListOf<Point>()

        for (id in targetIds) {
            val point3d = landmarks3d[id] ?: continue
            val point2d = landmarks2dById[id] ?: continue
            objectPoints.add(point3d)
            imagePoints.add(Point(point2d.x, point2d.y))
        }

        return poseEstimator.estimatePose(
            objectPoints = objectPoints,
            imagePoints = imagePoints,
            imageWidth = imageWidth,
            imageHeight = imageHeight
        )
    }
}
