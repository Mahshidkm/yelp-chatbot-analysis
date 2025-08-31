from flask import Blueprint, render_template, jsonify, request
from flask import current_app as app

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def landing():
    return render_template('index.html')
    
@main_bp.route('/restaurant-dashboard')
def restaurant_dashboard():
    restaurant_data = app.services.get_restaurant_data()
    return render_template('restaurant-dashboard.html', restaurants=restaurant_data)
    
@main_bp.route('/charts', methods=['POST'])
def charts():
    data = request.get_json()
    business_id = data['business_id']
    
    # Use the service instance properly
    result = app.services2.get_average_scores(business_id)
    
    return jsonify({
        'avg_scores': result["avg_scores"],
        'years': result["years"]
    })

@main_bp.route('/chat', methods=['GET', 'POST'])
def chat():
    # Load model only when chat route is accessed
    app.chat_model.load_chat_model()
    
    if request.method == 'POST':
        user_input = request.form['user_input']
        first_response = app.chat_model.process_user_input(user_input)
        post_response = app.chat_model.post_processed_response(first_response)
        bot_response = app.chat_model.analize_response(post_response)
        #bot_response ="hiiii"
        print(bot_response)
        #bot_response="i got your request"
        return jsonify({'response': bot_response})
        
    return render_template('chat.html')