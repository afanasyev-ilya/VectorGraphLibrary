from waitress import serve

import web_rating

serve(web_rating.app, host='0.0.0.0', port=80)