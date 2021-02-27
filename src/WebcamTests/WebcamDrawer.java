package WebcamTests;

import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;

public class WebcamDrawer extends JPanel{
	private Size size;
	private Mat image = new Mat(), img2 = new Mat();
	private boolean created = false;
	
	public WebcamDrawer(Size size) {
		this.size = size;
	}
	
	public void setMat(Mat mat) {
		this.image = mat;
		
		created = true;
	}
	
	public Mat getMat() {
		return image;
	}
	
	@Override
	public void paintComponent(Graphics g) {
		if(created) {
			Image img = HighGui.toBufferedImage(image);
			g.drawImage(img, 25, 25, null);
		}
	}
}
