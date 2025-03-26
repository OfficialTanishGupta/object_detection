package com.programminghut.object_detection

import android.content.Intent
import android.graphics.*
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.text.method.ScrollingMovementMethod
import android.view.MotionEvent
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import com.programminghut.object_detection.ml.SsdMobilenetV11Metadata1
import org.json.JSONObject
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    private val REQUEST_IMAGE_CAPTURE = 102
    private val REQUEST_IMAGE_PICK = 101
    private lateinit var currentPhotoPath: String

    lateinit var takePictureButton: Button
    lateinit var launchGalleryButton: Button
    lateinit var imageView: ImageView
    lateinit var resultText: TextView
    lateinit var photoURI: Uri
    lateinit var bitmap: Bitmap

    val paint = Paint()
    var detectedObjects = mutableListOf<DetectedObject>()
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
        resultText = findViewById(R.id.result)

        resultText.movementMethod = ScrollingMovementMethod()

        takePictureButton.setOnClickListener { dispatchTakePictureIntent() }
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
                    val file = File(currentPhotoPath)
                    bitmap = BitmapFactory.decodeFile(file.absolutePath)
                    imageView.setImageBitmap(bitmap)
                    getPredictions()
                }
            }
        }
    }

    private fun dispatchTakePictureIntent() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (intent.resolveActivity(packageManager) != null) {
            val photoFile: File? = try {
                createImageFile()
            } catch (ex: IOException) {
                null
            }
            if (photoFile != null) {
                photoURI = FileProvider.getUriForFile(
                    this,
                    "com.programminghut.object_detection.fileprovider",
                    photoFile
                )
                intent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(intent, REQUEST_IMAGE_CAPTURE)
            }
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String =
            SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault()).format(Date())
        val storageDir: File? = getExternalFilesDir(null)
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    private fun getPredictions() {
        detectedObjects.clear()
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
                val boundingBox = RectF(
                    locations[x + 1] * w, locations[x] * h,
                    locations[x + 3] * w, locations[x + 2] * h
                )
                val labelText = labels[classes[index].toInt()]
                detectedObjects.add(DetectedObject(boundingBox, labelText))

                paint.style = Paint.Style.STROKE
                paint.color = Color.RED
                canvas.drawRect(boundingBox, paint)
                paint.style = Paint.Style.FILL
                paint.color = Color.BLUE
                canvas.drawText(labelText, locations[x + 1] * w, locations[x] * h, paint)
            }
        }

        imageView.setImageBitmap(mutableBitmap)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        if (event.action == MotionEvent.ACTION_DOWN) {
            val x = event.x
            val y = event.y

            for (detectedObject in detectedObjects) {
                if (detectedObject.boundingBox.contains(x, y)) {
                    // Instead of opening a browser, fetch information and update the result TextView
                    fetchObjectInformation(detectedObject.label)
                    break
                }
            }
        }
        return super.onTouchEvent(event)
    }

    /**
     * Fetches a summary from Wikipedia for the given query and displays it in resultText.
     */
    private fun fetchObjectInformation(query: String) {
        Thread {
            try {
                // URL-encode the query string to handle spaces and special characters
                val encodedQuery = URLEncoder.encode(query, "UTF-8")
                // Wikipedia REST API for page summaries
                val urlStr = "https://en.wikipedia.org/api/rest_v1/page/summary/$encodedQuery"
                val url = URL(urlStr)
                val connection = url.openConnection() as HttpURLConnection
                connection.requestMethod = "GET"
                connection.connectTimeout = 5000
                connection.readTimeout = 5000
                val responseCode = connection.responseCode
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    val stream = connection.inputStream
                    val result = stream.bufferedReader().use { it.readText() }
                    // Parse the JSON response to extract the summary (the "extract" field)
                    val jsonObj = JSONObject(result)
                    val extract = jsonObj.optString("extract", "No information available.")
                    runOnUiThread {
                        resultText.text = extract
                    }
                } else {
                    runOnUiThread {
                        resultText.text = "Error: $responseCode"
                    }
                }
            } catch (e: Exception) {
                e.printStackTrace()
                runOnUiThread {
                    resultText.text = "Error fetching information."
                }
            }
        }.start()
    }
}

data class DetectedObject(val boundingBox: RectF, val label: String)
