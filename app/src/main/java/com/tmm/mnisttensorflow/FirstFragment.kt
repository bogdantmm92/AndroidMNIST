package com.tmm.mnisttensorflow

import android.graphics.PointF
import android.os.Bundle
import android.util.Log
import android.view.LayoutInflater
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.tmm.mnisttensorflow.databinding.FragmentFirstBinding
import com.tmm.mnisttensorflow.views.DrawModel
import java.io.File
import java.io.FileOutputStream
import java.util.Arrays

/**
 * A simple [Fragment] subclass as the default destination in the navigation.
 */
class FirstFragment : Fragment() {
    private var _binding: FragmentFirstBinding? = null

    private val PIXEL_WIDTH = 28

    private lateinit var drawModel: DrawModel
    private val mTmpPiont = PointF()
    private var mLastX = 0f
    private var mLastY = 0f

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

        drawModel = DrawModel(PIXEL_WIDTH, PIXEL_WIDTH)
        binding.drawView.setModel(drawModel)

        binding.btnClear.setOnClickListener {
            data.add(Arrays.toString(binding.drawView.pixelData))
            data.add("[${binding.label.text.toString()}]")
            saveFile(data)
            binding.label.text = ((binding.label.text.toString().toInt() + 1) % 10).toString()

            drawModel.clear()
            binding.drawView.reset()
            binding.drawView.invalidate()

            binding.tfRes.setText("")
        }

        binding.drawView.setOnTouchListener { view, event ->
            val action = event.action and MotionEvent.ACTION_MASK

            if (action == MotionEvent.ACTION_DOWN) {
                //begin drawing line
                processTouchDown(event)
                return@setOnTouchListener true
                //draw line in every direction the user moves
            } else if (action == MotionEvent.ACTION_MOVE) {
                processTouchMove(event)
                return@setOnTouchListener true
                //if finger is lifted, stop drawing
            } else if (action == MotionEvent.ACTION_UP) {
                processTouchUp()
                classify()
                return@setOnTouchListener true
            }
            return@setOnTouchListener false
        }
    }

    private val data = mutableListOf<String>()
    private fun classify() {
        digitClassifier.classifyAsync(binding.drawView.bitmap, binding.drawView.pixelData)
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

    override fun onResume() {
        binding.drawView.onResume()
        super.onResume()
    }

    override fun onPause() {
        binding.drawView.onPause()
        super.onPause()
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
    //draw line down
    private fun processTouchDown(event: MotionEvent) {
        //calculate the x, y coordinates where the user has touched
        mLastX = event.x
        mLastY = event.y
        //user them to calcualte the position
        binding.drawView.calcPos(mLastX, mLastY, mTmpPiont)
        //store them in memory to draw a line between the
        //difference in positions
        val lastConvX = mTmpPiont.x
        val lastConvY = mTmpPiont.y
        //and begin the line drawing
        drawModel.startLine(lastConvX, lastConvY)
    }

    //the main drawing function
    //it actually stores all the drawing positions
    //into the drawmodel object
    //we actually render the drawing from that object
    //in the drawrenderer class
    private fun processTouchMove(event: MotionEvent) {
        val x = event.x
        val y = event.y
        binding.drawView.calcPos(x, y, mTmpPiont)
        val newConvX = mTmpPiont.x
        val newConvY = mTmpPiont.y
        drawModel.addLineElem(newConvX, newConvY)
        mLastX = x
        mLastY = y
        binding.drawView.invalidate()
    }

    private fun processTouchUp() {
        drawModel.endLine()
    }
}