package com.example.facelandmarkar

/*
 * 描画側の考え方。
 *
 * 1. CameraXでカメラ画像を表示する。
 * 2. FacePosePipelineでPoseResultを求める。
 * 3. PoseResult.rotationMatrix と PoseResult.translationVector を
 *    OpenGL ESやSceneViewのモデル行列へ変換する。
 * 4. assetsに入れた face_model.glb などを読み込み、モデル行列を適用して描画する。
 *
 * Python版のようにOpenCV座標系とOpenGL座標系の違いがあるため、
 * PoseEstimator内でY軸とZ軸を反転している。
 *
 * Android実装では、まず以下を確認するとよい。
 *
 * - FaceMeshの2Dランドマークが正しくカメラ画像に重なるか。
 * - PoseResultのyaw/pitch/roll相当の値が顔の動きに合わせて変化するか。
 * - 固定モデルを表示できるか。
 * - 最後にR/tをモデル行列へ反映して顔へ重ねる。
 */
object RendererNotes
