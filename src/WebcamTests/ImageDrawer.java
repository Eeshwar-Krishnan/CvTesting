package WebcamTests;

import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.highgui.HighGui;

public class ImageDrawer extends JPanel {
	private Size size;
	private Image image;
	private boolean created = false;
	
	public ImageDrawer(Size size) {
		this.size = size;
	}
	
	public void setMat(Image mat) {
		this.image = mat;
		created = true;
	}
	
	@Override
	public void paintComponent(Graphics g) {
		if(created) {
			Image img = image;
			g.drawImage(img, 25, 25, null);
		}
	}
}
