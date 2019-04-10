package it.blog.tensorflow.bean;

public class Prediction {

	private String brand;
	private String percent;
	
	public Prediction(String brand, String percent)
	{
		this.brand = brand;
		this.percent = percent;
	}
	
	public String getBrand() {
		return brand;
	}
	public void setBrand(String brand) {
		this.brand = brand;
	}
	public String getPercent() {
		return percent;
	}
	public void setPercent(String percent) {
		this.percent = percent;
	}
	
	
}
