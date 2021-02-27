package ImageManipulation;

import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Image;

import javax.swing.JPanel;

import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;

public class MultiMatDrawer extends JPanel {
	private Mat one, two, three, four;
	private double fps;
	public MultiMatDrawer() {
		one = new Mat();
		two = new Mat();
		three = new Mat();
		four = new Mat();
		fps = 100.98;
	}
	
	public void setOne(Mat m) {
		this.one = m;
	}
	
	public void setTwo(Mat m) {
		this.two = m;
	}
	
	public void setThree(Mat m) {
		this.three = m;
	}
	
	public void setFour(Mat m) {
		this.four = m;
	}
	
	public void setFps(double d) {
		this.fps = d;
	}
	
	public void paintComponent(Graphics g) {
		int size = 2;
		if(one.size().width > 0 && two.size().width > 0 && three.size().width > 0 && four.size().width > 0) {
			Image i1 = HighGui.toBufferedImage(one);
			Image i2 = HighGui.toBufferedImage(two);
			Image i3 = HighGui.toBufferedImage(three);
			Image i4 = HighGui.toBufferedImage(four);
			g.drawImage(i1, 0, 0, one.width()/size, one.height()/size, null);
			g.drawImage(i2, one.width()/size, 0, one.width()/size, one.height()/size, null);
			g.drawImage(i3, 0, one.height()/size, one.width()/size, one.height()/size, null);
			g.drawImage(i4, one.width()/size, one.height()/size, one.width()/size, one.height()/size, null);
			g.setColor(Color.BLACK);
			g.fillRect(0, 0, 90, 30);
			if(fps <= 10) {
				g.setColor(Color.RED);
			}else if(fps <= 20) {
				g.setColor(Color.YELLOW);
			}else {
				g.setColor(Color.GREEN);
			}
			g.setFont(new Font("TimesRoman", Font.PLAIN, 30));
			g.drawString(String.format("%.2f", fps), 3, 25);
		}
	}
}
