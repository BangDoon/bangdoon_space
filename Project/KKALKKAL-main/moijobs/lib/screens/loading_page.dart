import 'dart:async';
import 'main_page.dart';
import 'package:flutter/material.dart';

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  //GetInformation getinformation = GetInformation();

  @override
  void initState() {
    super.initState();
    _init();
  }

  void _init() async {
    Timer(
      Duration(seconds: 2),
      () => Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (BuildContext context) => MainPage()),
        (route) => false,
      ),
    );

    // Navigator.pushAndRemoveUntil(
    //   context,
    //   MaterialPageRoute(builder: (BuildContext context) => MainPage()),
    //   (route) => false,
    // );

    // Navigator.push(
    //   context, MaterialPageRoute(builder: (context) => MainPage())
    // );
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        backgroundColor: Color(0xFF209FA6),
        body: Center(
          child: Container(
            decoration: BoxDecoration(
              gradient: LinearGradient(
                  begin: Alignment.topRight,
                  end: Alignment.bottomLeft,
                  colors: [Color(0xFF209FA6), Colors.white]),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Center(
                  child: Container(
                    width: 200,
                    height: 200,
                    child: Image(
                      image: AssetImage('images/appicon.png'),
                    ),
                  ),
                ),
                // CircularProgressIndicator(
                //   backgroundColor: Colors.white,
                //   strokeWidth: 6,
                // ),
                SizedBox(
                  height: 20,
                ),
                Text(
                  'loading...',
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.w500,
                    color: Colors.white,
                    shadows: [
                      Shadow(offset: Offset(4, 4), color: Colors.white10),
                    ],
                    decorationStyle: TextDecorationStyle.solid,
                  ),
                )
              ],
            ),
          ),
        ),
      ),
    );
  }
}
