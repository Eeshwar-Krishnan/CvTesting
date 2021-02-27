package WebcamTests;

import java.awt.BorderLayout;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.io.IOException;

import javax.swing.JFrame;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;
import org.opencv.videoio.VideoCapture;

public class WebcamTester extends WindowAdapter{
	
	private boolean finished;
	
	private static WebcamTester instance = new WebcamTester();

	public static void main(String[] args) throws IOException {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		instance.start();
		/**int camsTest = 750;
		for(int i = 0; i < camsTest; i ++) {
			VideoCapture capture = new VideoCapture(i);
			boolean success = capture.read(new Mat());
			if(success)
				System.out.println(i + " " + success);
		}
		*/
	}
	
	public void start() throws IOException{
		WebcamDrawer drawer = new WebcamDrawer(new Size(1280, 720));
		
		JFrame frame = new JFrame("Webcam");
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		
		frame.getContentPane().add(drawer, BorderLayout.CENTER);
		
		frame.repaint();
		frame.setSize(700, 600);
		frame.setVisible(true);
		
		VideoCapture capture = new VideoCapture(700);
		Mat image = new Mat();
		if(capture.isOpened()) {
			while(true) {
				capture.read(image);
				drawer.setMat(image);
				frame.repaint();
			}
		}
	}
	
    @Override
    public void windowClosed(WindowEvent e) {
    	finished = true;
    }
}
