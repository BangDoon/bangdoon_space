import 'package:flutter/material.dart';

import 'main_page.dart';


class ProfilePage extends StatelessWidget {

  @override
  Widget build(BuildContext context) {
    final double statusBarHeight = MediaQuery.of(context).padding.top;
    return Scaffold(
      body: Column(children: [
        Padding(
          padding: EdgeInsets.only(top: statusBarHeight),
        ),
        Padding(
          padding: const EdgeInsets.only(top: 50,left: 20,right: 20),
          child: Material(
            color: Color(0xFF3391E7),
            elevation: 5.0,
            borderRadius: BorderRadius.circular(20),
            child: Center(
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical:10),
                child: Text(
                  '프로필',
                  style: TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ),
        ),
        Row(),
        Expanded(child: Container()),
        SizedBox(),
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
          child: Material(
            borderRadius: BorderRadius.circular(20),
            elevation: 5,
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 10),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  FlatButton(
                    onPressed: () {
                      Navigator.pop(context);
                    },
                    child: Text(
                      '저장하고 닫기',
                      style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
        // SizedBox(height: 10,),
      ]),
    );
  }
}
