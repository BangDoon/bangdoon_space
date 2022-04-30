import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'profilepage.dart';
import 'reusable_card.dart';
import 'icon_content.dart';
import 'search_page.dart';

class MainPage extends StatefulWidget {
  @override
  _MainPageState createState() => _MainPageState();
}

class _MainPageState extends State<MainPage> {
  @override
  void initState() {
    // TODO: implement initState
    super.initState();
  }

  @override
  Widget build(BuildContext context) {
    final double statusBarHeight = MediaQuery.of(context).padding.top;
    return Scaffold(
      body: Column(
        children: [
          Container(
            decoration: BoxDecoration(
              color: Color(0xFF209FA6),
              borderRadius: BorderRadius.only(
                bottomLeft: Radius.circular(35),
                bottomRight: Radius.circular(35),
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: EdgeInsets.only(top: statusBarHeight),
                ),
                Row(
                  children: [
                    Padding(
                      padding: const EdgeInsets.only(
                        right: 30,
                      ),
                      child: Icon(null),
                    ),
                    Expanded(
                      child: Center(
                        child: Container(
                          height: 130,
                          width: 200,
                          child: Image(
                            image: AssetImage('images/appicon.png'),
                          ),
                        ),
                      ),
                    ),
                    Padding(
                      padding: const EdgeInsets.only(
                        right: 30,
                      ),
                      child: Material(
                        borderRadius: BorderRadius.circular(10),
                        elevation: 5.0,
                        color: Color(0xFFF5F9FF),
                        child: Container(
                          child: Padding(
                            padding: const EdgeInsets.all(8.0),
                            child: Icon(
                              Icons.notifications_none,
                              color: Color(0xFF1A202C),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                SizedBox(
                  height: 10,
                ),
                Row(
                  children: [
                    Expanded(
                      child: Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(20),
                        ),
                        padding: EdgeInsets.symmetric(
                          vertical: 10,
                        ),
                        margin: EdgeInsets.symmetric(
                          horizontal: 35,
                        ),
                        child: FlatButton(
                          onPressed: () {
                            Navigator.push(
                              context,
                              MaterialPageRoute(
                                builder: (BuildContext context) {
                                  return SearchExample();
                                },
                              ),
                            );
                            setState(() {});
                          },
                          child: Container(
                            child: Center(
                              child: Text(
                                '검색',
                                style: TextStyle(
                                  fontSize: 20,
                                  color: Color(0xFFCEDAF6),
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                SizedBox(
                  height: 30,
                ),
              ],
            ),
          ),
          Row(
            children: [
              Padding(
                padding: const EdgeInsets.only(left: 35, top: 20, bottom: 20),
                child: Text(
                  '게시글',
                  style: TextStyle(
                    fontSize: 25,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF1A202C),
                  ),
                ),
              ),
              Expanded(
                child: SizedBox(),
              ),
              Padding(
                padding: const EdgeInsets.only(top: 20, right: 10, bottom: 20),
                child: FlatButton(
                  child: Icon(Icons.refresh),
                  onPressed: () {
                    print('pressed');
                  },
                ),
              ),
            ],
          ),
          Expanded(
            child: Container(child: ReusableCard()
                // ListView(
                //   children: [
                //     ReusableCard(),
                //   ],
                // ),
                ),
          ),
          SizedBox(
            height: 20,
          ),
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 20),
            child: Material(
              borderRadius: BorderRadius.circular(20),
              elevation: 5.0,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  FlatButton(
                    onPressed: () {
                      setState(() {
                        print('pressed');
                      });
                    },
                    child: IconContent(
                      topicon: Icons.home_filled,
                      bottomicon: Icons.arrow_drop_up_rounded,
                      colour: Color(0xFF209FA6),
                    ),
                  ),
                  FlatButton(
                    onPressed: () {
                      setState(() {
                        print('pressed');
                      });
                    },
                    child: IconContent(
                      topicon: Icons.folder,
                      bottomicon: Icons.arrow_drop_up_rounded,
                      colour: Color(0xFFFA9124),
                    ),
                  ),
                  FlatButton(
                    onPressed: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (BuildContext context) {
                            return ProfilePage();
                          },
                        ),
                      );
                      setState(() {});
                    },
                    child: IconContent(
                      topicon: Icons.person_rounded,
                      bottomicon: Icons.arrow_drop_up_rounded,
                      colour: Color(0xFF3391E7),
                    ),
                  ),
                  FlatButton(
                    onPressed: () {
                      setState(() {
                        print('pressed');
                      });
                    },
                    child: IconContent(
                      topicon: Icons.settings,
                      bottomicon: Icons.arrow_drop_up_rounded,
                      colour: Color(0xFFFA5A7D),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
