from flask import Flask, render_template, request, send_from_directory
from generator import ImageGenerator
import os

app = Flask(__name__)
generator = ImageGenerator()


@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None

    if request.method == 'POST':
        prompt = request.form.get('prompt')
        model = request.form.get('model')
        steps = int(request.form.get('steps', 20))
        generator.config["default_steps"] = steps

        print(f"Generating: {prompt} with {model}...")
        result_paths = generator.generate_batch(model, prompt, num_images=1)

        if result_paths:
            full_path = result_paths[0]
            filename = os.path.basename(full_path)
            image_url = filename

    return render_template('index.html', image=image_url)

@app.route('/outputs/<filename>')
def uploaded_file(filename):
    return send_from_directory(generator.config['output_folder'], filename)

#if __name__ == '__main__':
    #app.run(debug=True, port=5000)
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=2222)
