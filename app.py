from flask import Flask
import requests
from forecast import *
from flask import Flask, Response, flash, redirect, render_template, request, session, abort, jsonify
import os
import json
import datetime
from copy import deepcopy
import base64
import uuid
from Map_Nav import Navigator
from pymongo import MongoClient
from bson.objectid import ObjectId
from bson.json_util import dumps
import urlparse
import psycopg2
import psycopg2.extras
from user_model import User
from flask_jwt_extended import (
    JWTManager, jwt_required, create_access_token,
    get_jwt_identity
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
engine = create_engine('sqlite:///usersV2.db', echo=True)
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY')
jwt = JWTManager(app)

def get_post_gres():
    # PostGreSQL fetch
    urlparse.uses_netloc.append("postgres")
    url = urlparse.urlparse(os.environ.get('POSTGRES_URL'))
    try:
        postgresql_conn = psycopg2.connect(
            database=url.path[1:],
            user=url.username,
            password=url.password,
            host=url.hostname,
            port=url.port
        )
    except:
        return jsonify({'Error_Message':"Failure to connect to the database"},
                       {'Content-Type': 'text/html'}, {'Status': 400})
    return postgresql_conn

@app.route('/login', methods=['POST'])
def user_login():
    POST_USERNAME = str(request.form['username'])
    POST_PASSWORD = str(request.form['password'])
    # print(POST_PASSWORD)
    Session = sessionmaker(bind=engine)
    s = Session()
    targ_user = s.query(User).filter_by(username=POST_USERNAME).one()

    if targ_user.check_password(POST_PASSWORD):
        session['logged_in'] = True
    else:
        flash('Invalid Username or Password; you will be redirected, please revise login credentials and reattempt')
    return home()

@app.route('/request_handler/admin/add_user', methods=['GET','POST'])
@jwt_required
def admin_user_add():
    post_gres_sql_conn = get_post_gres()
    cursor = post_gres_sql_conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    admin_identity = get_jwt_identity()
    query = 'SELECT username, password, role FROM bill_handling."users" WHERE username=%s' #% "'" + admin_identity + "'"
    cursor.execute(query, [admin_identity])
    admin_user = cursor.fetchone()

    if admin_user[2] != "Admin":
        return jsonify({"Error": "Non-admin users prohibited"}), 400
    else:
        username = request.args.get("Username")
        password = request.args.get("Password")
        role = request.args.get("Role")
        new_usr = User(username,password)
        query = 'INSERT INTO bill_handling."users" (username,id,password,role) VALUES (%s, %s, %s, %s);'
        cursor.execute(query,(new_usr.username,str(uuid.uuid4()),new_usr.pw_hash,role))
        post_gres_sql_conn.commit()
        cursor.close()
        post_gres_sql_conn.close()
        resp_msg = "User %s added"% str(new_usr.username)
        return jsonify({"Success": resp_msg}), 200

@app.route('/view_test_analysis',methods=['GET','POST'])
@jwt_required
def view_test_analysis():
    print("Ever After!")