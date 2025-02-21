package com.programminghut.object_detection

import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.programminghut.object_detection.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private val REQUEST_IMAGE_CAPTURE = 102
    private val REQUEST_IMAGE_PICK = 101

    lateinit var takePictureButton: Button
    lateinit var launchGalleryButton: Button
    lateinit var imageView: ImageView
    lateinit var classifiedText: TextView
    lateinit var resultText: TextView
    lateinit var photoURI: Uri
    lateinit var bitmap: Bitmap

    val paint = Paint()
    var colors = listOf(
        Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY,
        Color.BLACK, Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED
    )
    lateinit var labels: List<String>
    lateinit var model: SsdMobilenetV11Metadata1
    val imageProcessor = ImageProcessor.Builder()
        .add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR))
        .build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)

        takePictureButton = findViewById(R.id.button)
        launchGalleryButton = findViewById(R.id.button2)
        imageView = findViewById(R.id.imageView)
        classifiedText = findViewById(R.id.classified)
        resultText = findViewById(R.id.result)

        // Take Picture Button
        takePictureButton.setOnClickListener {
            dispatchTakePictureIntent()
        }

        // Launch Gallery Button
        launchGalleryButton.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, REQUEST_IMAGE_PICK)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (resultCode == RESULT_OK) {
            when (requestCode) {
                REQUEST_IMAGE_PICK -> {
                    val uri = data?.data
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
                    imageView.setImageBitmap(bitmap)
                    getPredictions()
                }
                REQUEST_IMAGE_CAPTURE -> {
                    bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, photoURI)
                    imageView.setImageBitmap(bitmap)
                    getPredictions()
                }
            }
        }
    }

    private fun dispatchTakePictureIntent() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        val photoFile: File? = createImageFile()
        photoFile?.let {
            photoURI = FileProvider.getUriForFile(this, "com.programminghut.object_detection.fileprovider", it)
            intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
        }
    }

    private fun createImageFile(): File? {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir = getExternalFilesDir(null)
        return File.createTempFile("JPEG_${timestamp}_", ".jpg", storageDir)
    }

    private fun getPredictions() {
        val image = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(image)
        val outputs = model.process(processedImage)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        val mutableBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutableBitmap)
        val h = mutableBitmap.height
        val w = mutableBitmap.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f

        scores.forEachIndexed { index, score ->
            if (score > 0.5) {
                val x = index * 4
                paint.color = colors[index % colors.size]
                paint.style = Paint.Style.STROKE
                canvas.drawRect(
                    RectF(
                        locations[x + 1] * w, locations[x] * h,
                        locations[x + 3] * w, locations[x + 2] * h
                    ), paint
                )
                paint.style = Paint.Style.FILL
                canvas.drawText(
                    "${labels[classes[index].toInt()]}: ${"%.2f".format(score)}",
                    locations[x + 1] * w, locations[x] * h, paint
                )
            }
        }

        imageView.setImageBitmap(mutableBitmap)
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}
