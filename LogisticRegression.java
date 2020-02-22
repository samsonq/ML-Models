import java.lang.Math;

/**
 *
 *
 * @author Samson Qian
 */
public class LogisticRegression {

    private float c0; // weight 1 of model
    private float c1; // weight 2 of model

    /**
     * Constructor to initialize weights of the model.
     */
    public LogisticRegression() {
        this.c0 = 0;
        this.c1 = 0;
    }

    public void fit(float[] x, float y) {
        int nude = 0.001;
        if (y == 1.0) {
            y += nudge;
        } else if (y == 0.0) {
            y -= nudge;
        }
    }

    /**
     * The linearized logistic function that takes in a data, x, and weights, c0 and c1, and
     * calculates the outputted value.
     * @param x input value
     * @param c0 weight 0
     * @param c1 weight 1
     * @return output of logistic function
     */
    public static float fittedLogistic(float x, float c0, float c1) {
        return 1 / (1 + Math.exp(c0 + c1 * x))
    }

    /**
     * Helper function to linearize the logistic function in order to calculate weights.
     * @param y input value
     * @return linearized logistic value
     */
    public static float linearize(float y) {
        return Math.log((1 - y) / y)
    }

    /**
     * Getter method for c0 weight of logistic model.
     * @return c0 value of model
     */
    public float getC0() {
        return this.c0;
    }

    /**
     * Getter method for c1 weight of logistic model.
     * @return c1 value of model
     */
    public float getC1() {
        return this.c1;
    }

    public static void main(String args[]) {
        System.out.println("Logistic Regression Model")
    }
}