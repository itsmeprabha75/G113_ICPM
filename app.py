from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import torch
from torchvision import transforms
from PIL import Image
import json
import os

# Import from training script
from train import DiseaseClassifier, LabelMapper

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['DATASET_PATH'] = 'tomato'
db = SQLAlchemy(app)

# Initialize label mapper with dataset path
label_mapper = LabelMapper(app.config['DATASET_PATH'])

# Disease treatments dictionary
TREATMENTS = {
    'Tomato___Bacterial_spot': {
        'description': 'Small, water-soaked lesions on leaves and fruits that turn brown, leading to defoliation and reduced yield.',
        'treatment': 'Use copper-based sprays and resistant varieties; remove infected plants and avoid working in the garden when plants are wet.'
    },
    'Tomato___Early_blight': {
        'description': 'Dark, concentric rings on leaves and stems, leading to yellowing and leaf drop. Affects older leaves first.',
        'treatment': 'Apply fungicides such as copper sprays or chlorothalonil. Use resistant varieties and practice crop rotation.'
    },
    'Tomato___healthy': {
        'description': 'Healthy plants without disease. Maintain proper care to prevent issues.',
        'treatment': 'Regularly monitor plants for early signs of disease. Ensure optimal soil, water, and nutrient management.'
    },
    'Tomato___Late_blight': {
        'description': 'Rapidly spreading disease causing water-soaked lesions on leaves and fruits, turning black and leading to plant death.',
        'treatment': 'Remove affected foliage and apply fungicides containing mancozeb or chlorothalonil. Avoid wet leaves by watering at the base.'
    },
    'Tomato___Leaf_Mold': {
        'description': 'Causes yellow spots on leaves, which turn brown with a velvety mold underneath. Common in humid environments.',
        'treatment': 'Improve air circulation, reduce humidity in greenhouses, and use fungicides like chlorothalonil for control.'
    },
    'Tomato___Septoria_leaf_spot': {
        'description': 'Causes small, dark spots with light centers on leaves, leading to leaf yellowing and drop. Reduces plant vigor.',
        'treatment': 'Remove and destroy infected leaves. Apply fungicides like copper or mancozeb, and ensure good air circulation and crop spacing.'
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'description': 'Tiny pests causing yellow stippling on leaves, fine webbing, and leaf drop. Thrive in hot, dry conditions.',
        'treatment': 'Spray horticultural oils or miticides. Encourage natural predators like ladybugs and maintain proper moisture levels.'
    },
    'Tomato___Target_Spot': {
        'description': 'Circular brown spots with concentric rings.',
        'treatment': 'Apply fungicides containing mancozeb. Improve air circulation.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'description': 'A viral disease that causes mottled and distorted leaves.',
        'treatment': 'Remove infected plants, use virus-free seeds, and disinfect tools.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'description': 'Causes yellowing, upward curling of leaves, and stunted plant growth. Spread by whiteflies.',
        'treatment': 'Control whiteflies using insecticides or sticky traps. Plant resistant varieties and use reflective mulches to deter insects.'
    }
}

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(120), nullable=False)

# Load the trained model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DiseaseClassifier(num_classes=len(label_mapper.label_to_idx)).to(DEVICE)

# Load the saved model
try:
    # Try loading the full checkpoint first
    checkpoint = torch.load('best_model.pth', map_location=DEVICE)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # If it's just the state dict
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

model.eval()

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    """Preprocess image for model prediction."""
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image.unsqueeze(0)

@app.route('/')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password_hash, request.form['password']):
            session['username'] = user.username
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
        
        user = User(
            username=username,
            password_hash=generate_password_hash(request.form['password'])
        )
        db.session.add(user)
        db.session.commit()
        flash('Registration successful')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        flash('No image uploaded')
        return redirect(url_for('home'))
    
    file = request.files['image']
    if file.filename == '':
        flash('No image selected')
        return redirect(url_for('home'))
    
    # Save and process the image
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(image_path)
    
    # Make prediction
    try:
        with torch.no_grad():
            image = preprocess_image(image_path).to(DEVICE)
            outputs = model(image)
            _, predicted = outputs.max(1)
        
        # Convert prediction to class name using label mapper
        predicted_class = label_mapper.idx_to_label[predicted.item()]
        
        # Get treatment information
        treatment_info = TREATMENTS[predicted_class]
        
        return render_template(
            'result.html',
            disease=predicted_class,
            description=treatment_info['description'],
            treatment=treatment_info['treatment'],
            image_path=image_path
        )
    except Exception as e:
        flash(f'Error during prediction: {str(e)}')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    # Ensure upload directory exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    with app.app_context():
        db.create_all()
    app.run(debug=True)