package com.example.facelandmarkar

import org.opencv.calib3d.Calib3d
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3

data class PoseResult(
    val rotationMatrix: Mat,
    val translationVector: Mat,
    val rotationVector: Mat
)

class PoseEstimator(
    private val focalLength: Double
) {
    fun estimatePose(
        objectPoints: List<Point3>,
        imagePoints: List<Point>,
        imageWidth: Int,
        imageHeight: Int
    ): PoseResult? {
        if (objectPoints.size < 6 || objectPoints.size != imagePoints.size) {
            return null
        }

        val cameraMatrix = Mat.eye(3, 3, CvType.CV_64F)
        cameraMatrix.put(0, 0, focalLength)
        cameraMatrix.put(0, 2, imageWidth / 2.0)
        cameraMatrix.put(1, 1, focalLength)
        cameraMatrix.put(1, 2, imageHeight / 2.0)

        val distCoeffs = MatOfDouble(0.0, 0.0, 0.0, 0.0)
        val obj = MatOfPoint3f(*objectPoints.toTypedArray())
        val img = MatOfPoint2f(*imagePoints.toTypedArray())

        val rvecInitial = Mat()
        val tvecInitial = Mat()

        val initialFlag = if (hasSqPnP()) {
            Calib3d.SOLVEPNP_SQPNP
        } else {
            Calib3d.SOLVEPNP_EPNP
        }

        val initialSuccess = Calib3d.solvePnP(
            obj,
            img,
            cameraMatrix,
            distCoeffs,
            rvecInitial,
            tvecInitial,
            false,
            initialFlag
        )
        if (!initialSuccess) {
            return null
        }

        val rvec = rvecInitial.clone()
        val tvec = tvecInitial.clone()
        val refinedSuccess = Calib3d.solvePnP(
            obj,
            img,
            cameraMatrix,
            distCoeffs,
            rvec,
            tvec,
            true,
            Calib3d.SOLVEPNP_ITERATIVE
        )
        if (!refinedSuccess) {
            return null
        }

        val rotation = Mat()
        Calib3d.Rodrigues(rvec, rotation)

        val axisTransform = Mat.eye(3, 3, CvType.CV_64F)
        axisTransform.put(1, 1, -1.0)
        axisTransform.put(2, 2, -1.0)

        val convertedRotation = Mat()
        val convertedTranslation = Mat()
        org.opencv.core.Core.gemm(axisTransform, rotation, 1.0, Mat(), 0.0, convertedRotation)
        org.opencv.core.Core.gemm(axisTransform, tvec, 1.0, Mat(), 0.0, convertedTranslation)

        return PoseResult(
            rotationMatrix = convertedRotation,
            translationVector = convertedTranslation,
            rotationVector = rvec
        )
    }

    private fun hasSqPnP(): Boolean {
        return try {
            Calib3d.SOLVEPNP_SQPNP
            true
        } catch (_: Throwable) {
            false
        }
    }
}
