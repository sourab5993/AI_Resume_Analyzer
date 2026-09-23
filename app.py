from main import app, matcher

# Expose WSGI application and handler for Vercel Serverless
handler = app

if __name__ == '__main__':
    app.run(debug=True)
