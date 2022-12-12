{
    "builds": [{
        "src": "yogaclasses/wsgi.py",
        "use": "@vercel/python",
        "config": { "maxLambdaSize": "15mb", "runtime":"python3.9" }
    }],
    "routes": [
        {
            "src": "build_files.sh",
            "dest": "vercel_app/wsgi.py"
        }
    ]
}