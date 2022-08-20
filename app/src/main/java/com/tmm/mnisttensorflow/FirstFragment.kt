package com.tmm.mnisttensorflow

import android.graphics.Color
import android.graphics.PointF
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.tmm.mnisttensorflow.databinding.FragmentFirstBinding
import java.io.File
import java.io.FileOutputStream
import java.util.Arrays

/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class FirstFragment : Fragment() {
    private var _binding: FragmentFirstBinding? = null

    private lateinit var digitClassifier: DigitClassifier

    private val binding get() = _binding!!
    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
        _binding = FragmentFirstBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        digitClassifier = DigitClassifier(requireContext())

        digitClassifier
            .initialize()
            .addOnFailureListener { e -> Log.e("yyy", "Error to setting up digit classifier.", e) }

        binding.drawView.setStrokeWidth(50.0f)
        binding.drawView.setColor(Color.WHITE)
        binding.drawView.setBackgroundColor(Color.BLACK)

        binding.drawView.setOnTouchListener { _, event ->
            // As we have interrupted DrawView's touch event,
            // we first need to pass touch events through to the instance for the drawing to show up
            binding.drawView.onTouchEvent(event)
            // Then if user finished a touch event, run classification
            if (event.action == MotionEvent.ACTION_UP) {
                classify()
            }
            true
        }
        binding.btnClear.setOnClickListener {
            data.add(Arrays.toString(digitClassifier
                .convertBitmapToPixels(
                    digitClassifier.getScaledBitmap(binding.drawView.getBitmap())
                )))
            data.add("[${binding.label.text.toString()}]")
            saveFile(data)
            binding.label.text = ((binding.label.text.toString().toInt() + 1) % 10).toString()

            binding.drawView.clearCanvas()

            binding.tfRes.setText("")
        }


    }

    private val data = mutableListOf<String>()
    private fun classify() {
        digitClassifier.classifyAsync(binding.drawView.getBitmap())
            .addOnSuccessListener { resultText ->
                binding.tfRes.text = resultText
            }
            .addOnFailureListener { e ->
                binding.tfRes.text = e.localizedMessage
                Log.e("yyy", "Error classifying drawing.", e)
            }
    }

    override fun onDestroyView() {
        digitClassifier.close()
        super.onDestroyView()
        _binding = null
    }

    private fun saveFile(list: MutableList<String>) {
        val path = requireContext().getExternalFilesDir(null)
        val file = File(path, "android_mnist_examples.json")
        val stream = FileOutputStream(file)
        try {
            stream.write(list.joinToString(",\n").toByteArray())
        } finally {
            stream.close()
        }
    }
}